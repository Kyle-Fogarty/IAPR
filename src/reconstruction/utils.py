"""
Utility functions for the RIMLS package.
"""
import torch
from typing import Optional


def compute_average_spacing_from_knn(knn_distances: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute average point spacing using the distance to the closest neighbor
    from precomputed KNN distances.

    Args:
        knn_distances: (N, K) tensor of distances to K nearest neighbors for N points.
                       Assumes K >= 1.
        eps: Small epsilon for numerical stability.

    Returns:
        float: Estimated average spacing.
    """
    if knn_distances.shape[1] == 0:
        # Handle case with no neighbors (e.g., single vertex)
        return eps
        
    # Use distance to the closest neighbor (knn_distances[:, 0] assuming sorted)
    closest_neighbor_dist = knn_distances[:, 0]
    return closest_neighbor_dist.mean().item()


import plyfile
import numpy as np






def loadply(filepath: str, device = 'cuda'):
    '''
    Loads pointcloud and normals from disk
    '''
    ply_data = plyfile.PlyData.read(filepath)
    vertex_element = ply_data['vertex']

    points = np.vstack([
    vertex_element['x'],
    vertex_element['y'],
    vertex_element['z']]).T

    normals = np.vstack([
        vertex_element['nx'],
        vertex_element['ny'],
        vertex_element['nz']
    ]).T

    points, normals = torch.from_numpy(points).to(device), torch.from_numpy(normals).to(device)


    # Check for NaN or Inf values in points and normals
    invalid_mask = torch.isnan(points).any(dim=1) | torch.isinf(points).any(dim=1) | \
                torch.isnan(normals).any(dim=1) | torch.isinf(normals).any(dim=1)

    # Count invalid points for reporting
    num_invalid = invalid_mask.sum().item()
    if num_invalid > 0:
        print(f"Found {num_invalid} points with NaN or Inf values. Removing them.")

    # Keep only valid points and normals
    valid_mask = ~invalid_mask
    points = points[valid_mask]
    normals = normals[valid_mask]
    return points, normals



# Create a grid of query points for marching cubes
def create_grid_for_marching_cubes(points, resolution=64, padding=0.1, verbose = False):
    """
    Create a grid of query points for marching cubes based on the bounding box of input points.
    
    Args:
        points: Tensor (N, 3) of input point cloud vertices
        resolution: Number of voxels along each dimension
        padding: Padding factor to extend the bounding box (as a fraction of the box size)
        
    Returns:
        grid_points: Tensor (resolution^3, 3) of query points in a 3D grid
        grid_shape: Tuple (resolution, resolution, resolution) for reshaping results
        min_bound: Tensor (3,) minimum coordinates of the grid
        max_bound: Tensor (3,) maximum coordinates of the grid
        voxel_size: Size of each voxel
    """
    # Compute bounding box
    min_bound, _ = torch.min(points, dim=0)
    max_bound, _ = torch.max(points, dim=0)
    
    # Add padding
    bbox_size = max_bound - min_bound
    padding_amount = bbox_size * padding
    min_bound = min_bound - padding_amount
    max_bound = max_bound + padding_amount
    
    # Create grid
    x = torch.linspace(min_bound[0], max_bound[0], resolution, device=points.device)
    y = torch.linspace(min_bound[1], max_bound[1], resolution, device=points.device)
    z = torch.linspace(min_bound[2], max_bound[2], resolution, device=points.device)
    
    # Create meshgrid
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    
    # Reshape to (resolution^3, 3)
    grid_points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
    
    # Calculate voxel size
    voxel_size = (max_bound - min_bound) / (resolution - 1)
    if verbose:
        print(f"Created grid with {grid_points.shape[0]} query points")
        print(f"Grid bounds: {min_bound} to {max_bound}")
        print(f"Voxel size: {voxel_size}")

    return grid_points, (resolution, resolution, resolution), min_bound, max_bound, voxel_size


def generate_mesh_from_sdf(sdf_grid, min_bound, max_bound, voxel_size, output_path, level=0.0, verbose=True):
    """
    Generate a mesh from an SDF grid using marching cubes and save it to disk.
    
    Args:
        sdf_grid (numpy.ndarray): 3D grid of SDF values
        min_bound (torch.Tensor or numpy.ndarray): Minimum coordinates of the grid
        max_bound (torch.Tensor or numpy.ndarray): Maximum coordinates of the grid
        voxel_size (torch.Tensor or numpy.ndarray): Size of each voxel
        output_path (str): Path where the mesh will be saved
        level (float): Isosurface level for marching cubes (default: 0.0)
        verbose (bool): Whether to print information about the mesh (default: True)
        
    Returns:
        trimesh.Trimesh: The generated mesh object
    """
    # Import necessary libraries for marching cubes
    from skimage import measure
    import trimesh
    import numpy as np
    import os
    
    # Convert torch tensors to numpy if needed
    if hasattr(min_bound, 'cpu'):
        min_bound = min_bound.cpu().numpy()
    if hasattr(max_bound, 'cpu'):
        max_bound = max_bound.cpu().numpy()
    if hasattr(voxel_size, 'cpu'):
        voxel_size = voxel_size.cpu().numpy()
    
    # Use marching cubes to generate mesh from SDF
    vertices, faces, normals, values = measure.marching_cubes(sdf_grid, level=level)

    # Scale vertices back to original coordinate system
    vertices = vertices * voxel_size + min_bound

    # Create a mesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the mesh to a file
    mesh.export(output_path)
    
    if verbose:
        print(f"Mesh generated and saved to {output_path}")
        print(f"Mesh contains {len(vertices)} vertices and {len(faces)} faces")
    
    return mesh


def process_grid_in_chunks(processor, grid_points, points, normals, 
                           chunk_size=100000, device=None, compute_gradients=True, 
                           verbose=True):
    """
    Process a large grid of points in chunks to compute SDF values and optionally gradients.
    
    Args:
        processor: The processor object that computes SDF values and gradients
        grid_points (torch.Tensor): Points on the grid to process (N, 3)
        points (torch.Tensor): Surface points (M, 3)
        normals (torch.Tensor): Surface normals (M, 3)
        chunk_size (int): Maximum number of points to process at once
        device (torch.device): Device to perform computation on
        compute_gradients (bool): Whether to compute and return gradients
        verbose (bool): Whether to print progress information
        
    Returns:
        tuple: (sdf, grad_sdf) where:
            - sdf (torch.Tensor): Signed distance function values (N,)
            - grad_sdf (torch.Tensor): Gradients of SDF (N, 3) or None if compute_gradients=False
    """
    if device is None:
        device = grid_points.device
    
    # Initialize tensors to store results
    num_points = grid_points.shape[0]
    num_chunks = (num_points + chunk_size - 1) // chunk_size  # Ceiling division
    
    sdf = torch.zeros(num_points, device=device)
    grad_sdf = torch.zeros(num_points, 3, device=device) if compute_gradients else None
    
    if verbose:
        print(f"Processing {num_points} grid points in {num_chunks} chunks of size {chunk_size}")
    
    # Process each chunk
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, num_points)
        
        if verbose:
            print(f"Processing chunk {i+1}/{num_chunks} ({start_idx} to {end_idx})")
        
        # Get current chunk
        chunk_points = grid_points[start_idx:end_idx]
        
        # Process chunk
        chunk_sdf, chunk_grad = processor(chunk_points, points, normals)
        
        # Store results
        sdf[start_idx:end_idx] = chunk_sdf
        if compute_gradients and chunk_grad is not None:
            grad_sdf[start_idx:end_idx] = chunk_grad
    
    return sdf, grad_sdf