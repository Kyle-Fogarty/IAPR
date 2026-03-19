"""
Robust Implicit Moving Least Squares (RIMLS) implementation with PyTorch.

This module provides the main RIMLS class for surface reconstruction from point clouds.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from .knn import find_knn_batched
from .utils import compute_average_spacing_from_knn


class RIMLS:
    """
    Robust Implicit Moving Least Squares (RIMLS) implementation with PyTorch.
    
    RIMLS is a technique used for surface reconstruction and processing from
    point clouds. It uses a robust statistical approach to estimate an 
    implicit surface.
    """
    
    def __init__(self,
                 sigma_n: float = 0.8,
                 sigma_r: float = 0.0,
                 refitting_threshold: float = 1e-3,
                 min_refitting_iters: int = 1,
                 max_refitting_iters: int = 3,
                 projection_accuracy_factor: float = 0.0001,
                 max_projection_iters: int = 15,
                 k_neighbors: int = 20,
                 knn_batch_size: int = 1024,
                 eps: float = 1e-8
                 ):
        """
        Initialize RIMLS with parameters.

        Args:
            sigma_n: Bandwidth for normal compatibility in refitting.
            sigma_r: (Currently unused) Regularization parameter.
            refitting_threshold: Convergence threshold for RIMLS refitting gradient change.
            min_refitting_iters: Minimum number of refitting iterations.
            max_refitting_iters: Maximum number of refitting iterations.
            projection_accuracy_factor: Factor multiplied by average spacing to get
                                       projection tolerance.
            max_projection_iters: Maximum number of projection steps.
            k_neighbors: Number of nearest neighbors to consider.
            knn_batch_size: Batch size for batched KNN calculation.
            eps: Small epsilon for numerical stability (divisions, normalization).
        """
        self.sigma_n = sigma_n
        self.sigma_r = sigma_r  # Note: Currently unused
        self.refitting_threshold = refitting_threshold
        self.min_refitting_iters = min_refitting_iters
        self.max_refitting_iters = max_refitting_iters
        self.projection_accuracy_factor = projection_accuracy_factor
        self.max_projection_iters = max_projection_iters
        self.k_neighbors = k_neighbors
        self.knn_batch_size = knn_batch_size
        self.eps = eps
    
    def compute_weights_and_derivatives(self, 
                                       query_points: torch.Tensor, 
                                       neighbors: torch.Tensor, 
                                       h: torch.Tensor
                                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute weights and their gradients for query points and neighbors.

        Args:
            query_points: (Q, 3) tensor of query positions.
            neighbors: (Q, K, 3) tensor of neighbor vertex positions.
            h: (Q,) tensor of bandwidth parameters for each query point.

        Returns:
            weights: (Q, K) tensor of weights.
            weight_gradients: (Q, K, 3) tensor of weight gradients w.r.t query_points.
            diffs: (Q, K, 3) tensor of (query_point - neighbor) differences.
        """
        # Compute differences and distances
        diffs = query_points.unsqueeze(1) - neighbors  # (Q, K, 3)
        distances_sq = torch.sum(diffs**2, dim=2)  # (Q, K)

        # Reshape h: (Q,) -> (Q, 1) for broadcasting
        h_sq = (h.unsqueeze(1))**2 + self.eps  # Add epsilon for stability
        
        weights = torch.exp(-distances_sq / h_sq)  # (Q, K)

        # Compute gradients (derivative w.r.t query_points)
        weight_gradients = -2.0 * weights.unsqueeze(-1) * diffs / h_sq.unsqueeze(-1)  # (Q, K, 3)

        return weights, weight_gradients, diffs
    
    def compute_potential_and_gradient(self, 
                                      x: torch.Tensor, 
                                      vertices: torch.Tensor, 
                                      normals: torch.Tensor, 
                                      neighborhood_indices: torch.Tensor, 
                                      h: torch.Tensor
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the potential and gradient at query points x.

        Args:
            x: (Q, 3) tensor of query positions.
            vertices: (N, 3) tensor of vertex positions.
            normals: (N, 3) tensor of vertex normals.
            neighborhood_indices: (Q, K) tensor of indices for K nearest neighbors.
            h: (Q,) tensor of bandwidth parameter for each query point.

        Returns:
            potential: (Q,) tensor of potential values.
            gradient: (Q, 3) tensor of gradients.
        """
        q_count = x.shape[0]
        k_count = neighborhood_indices.shape[1]
        device = x.device

        # Get neighbor positions and normals
        neighbors = vertices[neighborhood_indices]  # (Q, K, 3)
        neighbor_normals = normals[neighborhood_indices]  # (Q, K, 3)

        # Compute initial weights, gradients, and differences
        cached_weights, weight_gradients, diffs = self.compute_weights_and_derivatives(
            x, neighbors, h
        )
        
        # Initialize refitting weights
        cached_refitting_weights = torch.ones(q_count, k_count, device=device, dtype=x.dtype)
        
        grad = torch.zeros_like(x)

        for iter_count in range(self.max_refitting_iters):
            prev_grad = grad  # Save for convergence check

            # --- Start Refitting ---
            if iter_count > 0:
                # Compute refitting weights based on previous gradient (normal estimate)
                normal_diff_sq = torch.sum(
                    (neighbor_normals - prev_grad.unsqueeze(1))**2, dim=2
                )
                cached_refitting_weights = torch.exp(
                    -normal_diff_sq / (self.sigma_n**2 + self.eps)
                )
            # --- End Refitting ---

            # Compute total weights
            total_weights = cached_weights * cached_refitting_weights  # (Q, K)

            # Compute f = <x - p_i, n_i> (potential contribution)
            f = torch.sum(diffs * neighbor_normals, dim=2)  # (Q, K)

            # Compute weighted sums
            sum_w = total_weights.sum(dim=1) + self.eps  # (Q,) Add epsilon for stability
            sum_wf = (total_weights * f).sum(dim=1)  # (Q,)

            # Compute potential: P(x) = sum(w_i * f_i) / sum(w_i)
            potential = sum_wf / sum_w  # (Q,)

            # Compute gradient components
            weighted_gradients = weight_gradients * cached_refitting_weights.unsqueeze(-1)  # (Q, K, 3)
            sum_grad_w = weighted_gradients.sum(dim=1)  # (Q, 3)
            sum_grad_wf = (weighted_gradients * f.unsqueeze(-1)).sum(dim=1)  # (Q, 3)
            sum_wn = (total_weights.unsqueeze(-1) * neighbor_normals).sum(dim=1)  # (Q, 3)

            # Calculate gradient
            grad = (-sum_grad_w * potential.unsqueeze(-1) + sum_grad_wf + sum_wn) / sum_w.unsqueeze(-1)

            # Check convergence (compare new grad with prev_grad)
            if iter_count >= self.min_refitting_iters - 1:  # Check from min_iters onwards
                with torch.no_grad():
                    grad_diff_norm = torch.norm(grad - prev_grad, dim=1)
                    if torch.all(grad_diff_norm <= self.refitting_threshold):
                        break

        return potential, grad
    
    def compute_potential(self, 
                         points: torch.Tensor, 
                         vertices: torch.Tensor, 
                         normals: torch.Tensor
                         ) -> torch.Tensor:
        """
        Compute the potential value at query points.

        Args:
            points: (Q, 3) tensor of query points.
            vertices: (N, 3) tensor of vertex positions.
            normals: (N, 3) tensor of vertex normals.

        Returns:
            potential: (Q,) tensor of potential values.
        """
        k = min(self.k_neighbors, vertices.shape[0])
        if k == 0:
            return torch.zeros(points.shape[0], device=points.device, dtype=points.dtype)
            
        # 1. Find K nearest neighbors
        knn_distances, neighborhood_indices = find_knn_batched(
            points, vertices, k, batch_size=self.knn_batch_size
        )  # (Q, K), (Q, K)

        # 2. Compute local feature size (h) - using mean distance to K neighbors
        h = knn_distances.mean(dim=1) + self.eps  # (Q,)

        # 3. Compute potential and gradient
        potential, _ = self.compute_potential_and_gradient(
            points, vertices, normals, neighborhood_indices, h
        )
        return potential
    
    def project_points(self, 
                      points: torch.Tensor, 
                      vertices: torch.Tensor, 
                      normals: torch.Tensor
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project points onto the implicit surface defined by vertices and normals.

        Args:
            points: (Q, 3) tensor of points to project.
            vertices: (N, 3) tensor of vertex positions.
            normals: (N, 3) tensor of vertex normals.

        Returns:
            projected_points: (Q, 3) tensor of projected points.
            surface_normals: (Q, 3) tensor of surface normals at projected points.
        """
        projected = points.clone()
        q_count = points.shape[0]
        device = points.device
        dtype = points.dtype

        k = min(self.k_neighbors, vertices.shape[0])
        if k == 0:  # Handle edge case
            return projected, torch.zeros_like(projected)

        # 1. Initial KNN Search
        initial_knn_distances, initial_neighborhood_indices = find_knn_batched(
            points, vertices, k, batch_size=self.knn_batch_size
        )  # (Q, K), (Q, K)

        # 2. Estimate average spacing for projection tolerance
        average_spacing = compute_average_spacing_from_knn(initial_knn_distances, self.eps)
        projection_tolerance = average_spacing * self.projection_accuracy_factor
        
        # Initialize output normals
        output_normals = torch.zeros_like(points)

        # Projection iterations
        for iter_num in range(self.max_projection_iters):
            # Use initial KNN (fast approximation)
            neighborhood_indices = initial_neighborhood_indices
            
            # Compute local feature size (h) for current projected points
            current_neighbors = vertices[neighborhood_indices]  # (Q, K, 3)
            current_diffs_sq = torch.sum(
                (projected.unsqueeze(1) - current_neighbors)**2, dim=2
            )  # (Q,K)
            current_distances = torch.sqrt(current_diffs_sq)
            h = current_distances.mean(dim=1) + self.eps  # (Q,)

            # Compute potential and gradient at the current projected position
            potential, gradient = self.compute_potential_and_gradient(
                projected, vertices, normals, neighborhood_indices, h
            )  # (Q,), (Q, 3)

            # Normalize gradient to get normal vector
            gradient_norm = torch.linalg.norm(gradient, dim=1, keepdim=True) + self.eps
            normal = gradient / gradient_norm
            output_normals = normal  # Store the latest normal estimate

            # Compute update step
            delta = potential.unsqueeze(-1) * normal  # (Q, 1) * (Q, 3) -> (Q, 3)

            # Update projected points
            projected = projected - delta

            # Check convergence: Check if potential is close to zero
            with torch.no_grad():
                max_potential_abs = torch.max(torch.abs(potential))
                if max_potential_abs <= projection_tolerance:
                    break

        return projected, output_normals