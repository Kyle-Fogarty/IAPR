"""
K-Nearest Neighbors implementation optimized for 3D point clouds with PyTorch.
"""
import torch
from typing import Tuple


@torch.no_grad()
def find_knn_batched(query_points: torch.Tensor, 
                    vertices: torch.Tensor, 
                    k: int, 
                    batch_size: int = 1024
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Finds K nearest neighbors using batched distance computation for memory efficiency.

    Args:
        query_points: (Q, 3) tensor of query points.
        vertices: (N, 3) tensor of data points (point cloud).
        k: Number of neighbors to find.
        batch_size: Batch size for query points to manage memory.

    Returns:
        distances: (Q, k) tensor of distances to k nearest neighbors.
        indices: (Q, k) tensor of indices of k nearest neighbors.
    """
    q_count = query_points.shape[0]
    n_count = vertices.shape[0]
    device = query_points.device
    
    all_distances = torch.empty(q_count, k, device=device, dtype=query_points.dtype)
    all_indices = torch.empty(q_count, k, device=device, dtype=torch.long)

    k = min(k, n_count)  # Ensure k is not larger than the number of vertices

    for i in range(0, q_count, batch_size):
        start = i
        end = min(i + batch_size, q_count)
        batch_query = query_points[start:end]  # (B, 3)
        
        # Compute distances for the batch: (B, 1, 3) - (1, N, 3) -> (B, N, 3) -> (B, N)
        batch_diffs = batch_query.unsqueeze(1) - vertices.unsqueeze(0)
        batch_dists_sq = torch.sum(batch_diffs**2, dim=2)  # Use squared distances for topk, sqrt later
        
        # Find topk smallest distances
        topk_distances_sq, topk_indices = torch.topk(
            batch_dists_sq, k=k, dim=1, largest=False, sorted=True
        )
        
        all_distances[start:end] = torch.sqrt(topk_distances_sq)
        all_indices[start:end] = topk_indices
        
        # Free memory explicitly
        del batch_diffs, batch_dists_sq, topk_distances_sq, topk_indices

    return all_distances, all_indices