"""
Differentiable core RIMLS computation.

This module provides a PyTorch module that implements the core RIMLS computation
in a fully differentiable manner.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


class DifferentiableRIMLSCore(torch.nn.Module):
    """
    Computes RIMLS potential and/or gradient differentiably,
    assuming fixed neighbor indices and bandwidth.
    
    This class maintains the gradient chain through the RIMLS computation,
    allowing backpropagation through the potential and gradient calculations.
    """
    
    def __init__(self,
                 sigma_n: float = 0.8,
                 refitting_threshold: float = 1e-3,
                 min_refitting_iters: int = 1,
                 max_refitting_iters: int = 3,
                 eps: float = 1e-8):
        """
        Initialize the differentiable RIMLS core.
        
        Args:
            sigma_n: Bandwidth for normal compatibility in refitting
            refitting_threshold: Convergence threshold for refitting
            min_refitting_iters: Minimum number of refitting iterations
            max_refitting_iters: Maximum number of refitting iterations
            eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.sigma_n = sigma_n
        self.refitting_threshold = refitting_threshold
        self.min_refitting_iters = min_refitting_iters
        self.max_refitting_iters = max_refitting_iters
        self.eps = eps
    
    def _compute_weights_and_derivatives(self,
                                        query_points: torch.Tensor,
                                        neighbors: torch.Tensor,
                                        h: torch.Tensor
                                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute weights and their gradients for the given query points and neighbors.
        
        Args:
            query_points: (Q, 3) tensor of query positions
            neighbors: (Q, K, 3) tensor of neighbor vertex positions
            h: (Q,) or (Q, 1) tensor of bandwidth parameters
            
        Returns:
            weights: (Q, K) tensor of weights
            weight_gradients: (Q, K, 3) tensor of weight gradients w.r.t query_points
            diffs: (Q, K, 3) tensor of (query_point - neighbor) differences
        """
        # Ensure h has the right shape for broadcasting
        if h.ndim == 1:
            h = h.unsqueeze(1)  # (Q,) -> (Q, 1)
        
        # Compute differences and distances
        diffs = query_points.unsqueeze(1) - neighbors  # (Q, K, 3)
        distances_sq = torch.sum(diffs**2, dim=2)  # (Q, K)
        
        # Compute weights with Gaussian kernel
        h_sq = h**2 + self.eps  # Add epsilon for stability
        weights = torch.exp(-distances_sq / h_sq)  # (Q, K)
        
        # Compute gradients (derivative w.r.t query_points)
        weight_gradients = -2.0 * weights.unsqueeze(-1) * diffs / h_sq.unsqueeze(-1)  # (Q, K, 3)
        
        return weights, weight_gradients, diffs
    
    def forward(self,
               query_points: torch.Tensor,
               source_vertices: torch.Tensor,
               source_normals: torch.Tensor,
               neighbor_indices: torch.Tensor,
               bandwidth_h: torch.Tensor,
               compute_gradient: bool = False
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Differentiable forward pass for RIMLS potential/gradient calculation.
        
        Args:
            query_points: (Q, 3) tensor of query positions
            source_vertices: (N, 3) tensor of vertex positions
            source_normals: (N, 3) tensor of vertex normals
            neighbor_indices: (Q, K) tensor of indices for K nearest neighbors
            bandwidth_h: (Q,) tensor of bandwidth parameter for each query point
            compute_gradient: If True, also compute and return the gradient
            
        Returns:
            potential: (Q,) tensor of potential values
            gradient: (Q, 3) tensor of gradients, or None if compute_gradient is False
        """
        q_count = query_points.shape[0]
        k_count = neighbor_indices.shape[1]
        device = query_points.device
        dtype = query_points.dtype
        
        if k_count == 0:  # Handle cases with no neighbors
            potential = torch.zeros(q_count, device=device, dtype=dtype)
            gradient = torch.zeros(q_count, 3, device=device, dtype=dtype) if compute_gradient else None
            return potential, gradient
        
        # Gather neighbors (differentiable operation)
        safe_indices = torch.clamp(neighbor_indices, 0, source_vertices.shape[0]-1)
        neighbors = source_vertices[safe_indices]  # (Q, K, 3)
        neighbor_normals = source_normals[safe_indices]  # (Q, K, 3)
        
        # Initial weight computation
        h_unsqueezed = bandwidth_h.unsqueeze(1) if bandwidth_h.ndim == 1 else bandwidth_h
        base_weights, base_weight_grads, diffs = self._compute_weights_and_derivatives(
            query_points, neighbors, h_unsqueezed
        )
        
        # RIMLS refitting loop
        current_grad_estimate = torch.zeros(q_count, 3, device=device, dtype=dtype)
        sigma_n_sq = self.sigma_n**2 + self.eps
        
        for iter_count in range(self.max_refitting_iters):
            prev_grad_estimate_for_check = current_grad_estimate.clone()
            
            # Compute refitting weights based on normal compatibility
            if iter_count == 0:
                refitting_weights = torch.ones(q_count, k_count, device=device, dtype=dtype)
            else:
                normal_diff_sq = torch.sum(
                    (neighbor_normals.float() - current_grad_estimate.unsqueeze(1).float())**2, 
                    dim=2
                )
                refitting_weights = torch.exp(-normal_diff_sq / sigma_n_sq).to(dtype=dtype)
            
            # Compute total weights
            total_weights = base_weights * refitting_weights  # (Q, K)
            
            # Compute potential contribution from each neighbor
            f = torch.sum(diffs.float() * neighbor_normals.float(), dim=2).to(dtype=dtype)  # (Q, K)
            
            # Compute weighted sums
            sum_w = total_weights.sum(dim=1) + self.eps  # (Q,)
            sum_wf = (total_weights * f).sum(dim=1)  # (Q,)
            
            # Compute potential
            potential = sum_wf / sum_w  # (Q,)
            
            # Compute gradient components
            weighted_gradients = base_weight_grads * refitting_weights.unsqueeze(-1)  # (Q, K, 3)
            sum_grad_w = weighted_gradients.sum(dim=1)  # (Q, 3)
            sum_grad_wf = (weighted_gradients.float() * f.unsqueeze(-1).float()).sum(dim=1).to(dtype=dtype)  # (Q, 3)
            sum_wn = (total_weights.unsqueeze(-1).float() * neighbor_normals.float()).sum(dim=1).to(dtype=dtype)  # (Q, 3)
            
            # Compute gradient
            current_grad_estimate = (
                (-sum_grad_w.float() * potential.unsqueeze(-1).float() + 
                 sum_grad_wf.float() + sum_wn.float()) / 
                sum_w.unsqueeze(-1).float()
            ).to(dtype=dtype)
            
            # Check convergence
            if iter_count >= self.min_refitting_iters - 1:
                with torch.no_grad():
                    grad_diff_norm = torch.linalg.norm(
                        current_grad_estimate.float() - prev_grad_estimate_for_check.float(), 
                        dim=1
                    )
                    if torch.all(grad_diff_norm <= self.refitting_threshold):
                        break
        
        # Return potential and optionally gradient
        gradient_out = current_grad_estimate if compute_gradient else None
        return potential, gradient_out