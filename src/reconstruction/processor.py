"""
High-level RIMLS processor with Faiss acceleration.

This module provides a PyTorch module that implements RIMLS with accelerated
KNN using Faiss, while maintaining differentiability for the RIMLS computation.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

from .faiss_knn import FAISS_AVAILABLE, FaissKNN
from .differentiable import DifferentiableRIMLSCore


class RIMLSProcessor(torch.nn.Module):
    """
    Computes RIMLS potential and gradient on query points using a source point cloud.
    
    This processor uses Faiss for efficient KNN computation and maintains
    differentiability for the RIMLS potential and gradient calculations.
    """
    
    def __init__(self,
                 k_neighbors: int = 256,
                 sigma_n: float = 0.8,
                 refitting_threshold: float = 1e-3,
                 min_refitting_iters: int = 1,
                 max_refitting_iters: int = 1,
                 device_preference: str = 'cuda',
                 verbose: bool = True,
                 eps: float = 1e-8):
        """
        Initialize the RIMLS Processor.
        
        Args:
            k_neighbors: Number of nearest neighbors for RIMLS calculation
            sigma_n: RIMLS parameter for normal compatibility weighting during refitting
            refitting_threshold: Convergence threshold for RIMLS refitting
            min_refitting_iters: Minimum refitting iterations
            max_refitting_iters: Maximum refitting iterations
            device_preference: Preferred compute device ('cuda' or 'cpu')
            verbose: If True, print status messages
            eps: Small epsilon value for numerical stability
        """
        super().__init__()
        
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss library is required but not found or failed to import.")
        
        self.k_neighbors = k_neighbors
        self.device_preference = device_preference
        self.verbose = verbose
        self.eps = eps
        
        # Differentiable RIMLS core
        self.rimls_core = DifferentiableRIMLSCore(
            sigma_n=sigma_n,
            refitting_threshold=refitting_threshold,
            min_refitting_iters=min_refitting_iters,
            max_refitting_iters=max_refitting_iters,
            eps=eps
        )
        
        # Faiss KNN searcher
        self.knn = FaissKNN(
            device_preference=device_preference,
            verbose=verbose
        )
        
        # Device determination
        self.device = self.knn.device
        
        # Move the RIMLS core to the correct device
        self.rimls_core.to(self.device)
    
    def build_index(self, source_vertices: torch.Tensor, force_rebuild: bool = False) -> None:
        """
        Build or rebuild the Faiss index for the given source vertices.
        Must be called before forward pass, or whenever source vertices change significantly.
        
        Args:
            source_vertices: Tensor (M, 3) of source point cloud vertices
            force_rebuild: If True, always rebuild the index even if vertices seem unchanged
        """
        self.knn.build_index(source_vertices, force_rebuild)
    
    def forward(self,
               query_points: torch.Tensor,
               source_vertices: torch.Tensor,
               source_normals: torch.Tensor,
               compute_gradient: bool = False, 
               bandwidth = None 
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute RIMLS potential and optionally gradient on query_points.
        
        Requires `build_index(source_vertices)` to be called beforehand.
        
        Args:
            query_points: Tensor (N, 3) of points to evaluate potential at
            source_vertices: Tensor (M, 3) of source point cloud vertices
            source_normals: Tensor (M, 3) of source point cloud normals
            compute_gradient: If True, also compute and return the gradient
            
        Returns:
            potential: (N,) tensor of potential values
            gradient: (N, 3) tensor of gradients, or None if compute_gradient is False
        """
        # Input validation
        if self.knn._faiss_index is None:
            raise RuntimeError("Faiss index is not built. Call build_index(source_vertices) before calling forward.")
        
        if not isinstance(query_points, torch.Tensor) or query_points.ndim != 2 or query_points.shape[1] != 3:
            raise ValueError(f"query_points must be a Tensor of shape (N, 3), got {query_points.shape}")
        
        if not isinstance(source_vertices, torch.Tensor) or source_vertices.ndim != 2 or source_vertices.shape[1] != 3:
            raise ValueError(f"source_vertices must be a Tensor of shape (M, 3), got {source_vertices.shape}")
        
        if not isinstance(source_normals, torch.Tensor) or source_normals.shape != source_vertices.shape:
            raise ValueError(f"source_normals must be a Tensor with the same shape as source_vertices, got {source_normals.shape}")
        
        # Ensure inputs are on the correct device
        query_points = query_points.to(self.device)
        source_vertices = source_vertices.to(self.device)
        source_normals = source_normals.to(self.device)
        
        # Ensure normals are normalized (differentiable operation)
        source_normals = F.normalize(source_normals, p=2, dim=1, eps=1e-8)
        
        # Determine effective k based on the number of available points
        n_index_points = self.knn._faiss_index.ntotal
        effective_k = min(self.k_neighbors, n_index_points)
        
        if effective_k < self.k_neighbors and self.verbose:
            print(f"RIMLSProcessor: Warning: effective k reduced to {effective_k} due to index size ({n_index_points}).")
        
        if effective_k == 0:  # Handle case where k=0 or index is empty
            q_count = query_points.shape[0]
            dtype = query_points.dtype
            potential = torch.zeros(q_count, device=self.device, dtype=dtype)
            gradient = torch.zeros(q_count, 3, device=self.device, dtype=dtype) if compute_gradient else None
            return potential, gradient
        
        
        # Non-Differentiable KNN Search and Bandwidth Calculation
        with torch.no_grad():
            # Find k-nearest neighbors using Faiss
            knn_distances, neighbor_indices = self.knn.search(
                query_points, effective_k
            )
            
            # Calculate bandwidth 'h' (local feature size estimate)
            if knn_distances.numel() > 0 and knn_distances.shape[1] > 0:
                bandwidth_h =   knn_distances.mean(dim=1).detach() + self.eps
                # print(bandwidth_h.shape)
                # input()
                # print(f'bandwidth_h {bandwidth_h}')

            else:  # Fallback if no distances returned
                bandwidth_h = torch.full(
                    (query_points.shape[0],), 
                    self.eps, 
                    device=self.device, 
                    dtype=query_points.dtype
                )
        if bandwidth!= None:
            bandwidth_h = bandwidth_h + bandwidth
                    # print(f'bandwidth_h {bandwidth_h}')
        # Differentiable RIMLS Computation
        potential, gradient = self.rimls_core(
            query_points,        # Original tensor (can require grad)
            source_vertices,     # Original tensor (can require grad)
            source_normals,      # Original tensor (can require grad)
            neighbor_indices,    # Computed indices (no grad needed)
            bandwidth_h,         # Computed bandwidth (no grad needed)
            compute_gradient=compute_gradient
        )
        
        return potential, gradient
    
    def release_resources(self) -> None:
        """
        Explicitly release Faiss resources.
        Call this method when you're done using the processor to free memory.
        """
        self.knn.release_resources()
    
    def __del__(self):
        """Release resources when object is deleted."""
        self.release_resources()





class RIMLSAttNProcessor(torch.nn.Module):
    """
    Computes RIMLS potential and gradient on query points using a source point cloud.
    
    This processor uses Faiss for efficient KNN computation and maintains
    differentiability for the RIMLS potential and gradient calculations.
    """
    
    def __init__(self,
                 k_neighbors: int = 30,
                 sigma_n: float = 0.8,
                 refitting_threshold: float = 1e-3,
                 min_refitting_iters: int = 1,
                 max_refitting_iters: int = 3,
                 device_preference: str = 'cuda',
                 verbose: bool = True,
                 eps: float = 1e-8):
        """
        Initialize the RIMLS Processor.
        
        Args:
            k_neighbors: Number of nearest neighbors for RIMLS calculation
            sigma_n: RIMLS parameter for normal compatibility weighting during refitting
            refitting_threshold: Convergence threshold for RIMLS refitting
            min_refitting_iters: Minimum refitting iterations
            max_refitting_iters: Maximum refitting iterations
            device_preference: Preferred compute device ('cuda' or 'cpu')
            verbose: If True, print status messages
            eps: Small epsilon value for numerical stability
        """
        super().__init__()
        
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss library is required but not found or failed to import.")
        
        self.k_neighbors = k_neighbors
        self.device_preference = device_preference
        self.verbose = verbose
        self.eps = eps
        
        # Differentiable RIMLS core
        self.rimls_core = DifferentiableRIMLSCore(
            sigma_n=sigma_n,
            refitting_threshold=refitting_threshold,
            min_refitting_iters=min_refitting_iters,
            max_refitting_iters=max_refitting_iters,
            eps=eps
        )
        
        # Faiss KNN searcher
        self.knn = FaissKNN(
            device_preference=device_preference,
            verbose=verbose
        )
        
        # Device determination
        self.device = self.knn.device
        
        # Move the RIMLS core to the correct device
        self.rimls_core.to(self.device)
    
    def build_index(self, source_vertices: torch.Tensor, force_rebuild: bool = False) -> None:
        """
        Build or rebuild the Faiss index for the given source vertices.
        Must be called before forward pass, or whenever source vertices change significantly.
        
        Args:
            source_vertices: Tensor (M, 3) of source point cloud vertices
            force_rebuild: If True, always rebuild the index even if vertices seem unchanged
        """
        self.knn.build_index(source_vertices, force_rebuild)
    
    def forward(self,
               query_points: torch.Tensor,
               source_vertices: torch.Tensor,
               source_normals: torch.Tensor,
               compute_gradient: bool = False
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute RIMLS potential and optionally gradient on query_points.
        
        Requires `build_index(source_vertices)` to be called beforehand.
        
        Args:
            query_points: Tensor (N, 3) of points to evaluate potential at
            source_vertices: Tensor (M, 3) of source point cloud vertices
            source_normals: Tensor (M, 3) of source point cloud normals
            compute_gradient: If True, also compute and return the gradient
            
        Returns:
            potential: (N,) tensor of potential values
            gradient: (N, 3) tensor of gradients, or None if compute_gradient is False
        """
        # Input validation
        if self.knn._faiss_index is None:
            raise RuntimeError("Faiss index is not built. Call build_index(source_vertices) before calling forward.")
        
        if not isinstance(query_points, torch.Tensor) or query_points.ndim != 2 or query_points.shape[1] != 3:
            raise ValueError(f"query_points must be a Tensor of shape (N, 3), got {query_points.shape}")
        
        if not isinstance(source_vertices, torch.Tensor) or source_vertices.ndim != 2 or source_vertices.shape[1] != 3:
            raise ValueError(f"source_vertices must be a Tensor of shape (M, 3), got {source_vertices.shape}")
        
        if not isinstance(source_normals, torch.Tensor) or source_normals.shape != source_vertices.shape:
            raise ValueError(f"source_normals must be a Tensor with the same shape as source_vertices, got {source_normals.shape}")
        
        # Ensure inputs are on the correct device
        query_points = query_points.to(self.device)
        source_vertices = source_vertices.to(self.device)
        source_normals = source_normals.to(self.device)
        
        # Ensure normals are normalized (differentiable operation)
        source_normals = F.normalize(source_normals, p=2, dim=1, eps=1e-8)
        
        # Determine effective k based on the number of available points
        n_index_points = self.knn._faiss_index.ntotal
        effective_k = min(self.k_neighbors, n_index_points)
        
        if effective_k < self.k_neighbors and self.verbose:
            print(f"RIMLSProcessor: Warning: effective k reduced to {effective_k} due to index size ({n_index_points}).")
        
        if effective_k == 0:  # Handle case where k=0 or index is empty
            q_count = query_points.shape[0]
            dtype = query_points.dtype
            potential = torch.zeros(q_count, device=self.device, dtype=dtype)
            gradient = torch.zeros(q_count, 3, device=self.device, dtype=dtype) if compute_gradient else None
            return potential, gradient
        
        # Non-Differentiable KNN Search and Bandwidth Calculation
        with torch.no_grad():
            # Find k-nearest neighbors using Faiss
            knn_distances, neighbor_indices = self.knn.search(
                query_points, effective_k
            )
            
            # Calculate bandwidth 'h' (local feature size estimate)
            if knn_distances.numel() > 0 and knn_distances.shape[1] > 0:
                bandwidth_h =  knn_distances.mean(dim=1).detach() + self.eps
            else:  # Fallback if no distances returned
                bandwidth_h = torch.full(
                    (query_points.shape[0],), 
                    self.eps, 
                    device=self.device, 
                    dtype=query_points.dtype
                )
        
        # Differentiable RIMLS Computation
        potential, gradient = self.rimls_core(
            query_points,        # Original tensor (can require grad)
            source_vertices,     # Original tensor (can require grad)
            source_normals,      # Original tensor (can require grad)
            neighbor_indices,    # Computed indices (no grad needed)
            bandwidth_h,         # Computed bandwidth (no grad needed)
            compute_gradient=compute_gradient
        )
        
        return potential, gradient
    
    def release_resources(self) -> None:
        """
        Explicitly release Faiss resources.
        Call this method when you're done using the processor to free memory.
        """
        self.knn.release_resources()
    
    def __del__(self):
        """Release resources when object is deleted."""
        self.release_resources()