import torch 
import trimesh 
import numpy as np
import point_cloud_utils as pcu

def load_mesh_data(mesh_path, num_input_points, device):
    """Loads mesh, samples surface points, and gets bounds."""
    print(f"Loading mesh from {mesh_path}...")
    try:
        print(mesh_path)
        mesh = trimesh.load(mesh_path, process=False) # process=False is often safer
        # Ensure manifold if needed by subsequent operations, but sampling works without it
        # if not mesh.is_watertight:
        #     print("Warning: Mesh is not watertight.")
        # if not mesh.is_winding_consistent:
        #      print("Warning: Mesh winding is not consistent.")

    except Exception as e:
        print(f"Error loading mesh: {e}")
        raise

    print("Sampling surface points...")
    points_surface = mesh.sample(num_input_points)
    points_surface = torch.from_numpy(points_surface).float().unsqueeze(0).to(device) # [1, N_input, 3]

    # Prepare mesh data for pcu (use float64 as pcu often prefers it)
    vertices_np_f64 = mesh.vertices.astype(np.float64)
    faces_np = mesh.faces

    # Calculate bounds for query point sampling and mesh extraction
    bounds = np.array(mesh.bounds)
    center = torch.tensor(bounds.mean(axis=0), dtype=torch.float32, device=device)
    scale = torch.tensor((bounds[1] - bounds[0]).max(), dtype=torch.float32, device=device) # Use max extent for normalization

    print(f"Mesh loaded: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces.")
    print(f"Bounds: {bounds.tolist()}")

    return mesh, points_surface, vertices_np_f64, faces_np, bounds


def sample_query_points(num_points, bounds, batch_size, device):
    """Samples random query points within the mesh bounds."""
    bounds_min = torch.tensor(bounds[0], dtype=torch.float32, device=device)
    bounds_max = torch.tensor(bounds[1], dtype=torch.float32, device=device)
    # Sample within the bounding box
    query_points = torch.rand(batch_size, num_points, 3, device=device)
    query_points = query_points * (bounds_max - bounds_min) + bounds_min
    return query_points.float()


def calculate_gt_sdf(query_points_gpu, vertices_np_f64, faces_np, device):
    """Calculates ground truth SDF using PCU."""
    query_points_np = query_points_gpu.squeeze(0).detach().cpu().numpy().astype(np.float64) # [N_query, 3]

    sdf_gt_np, _, _ = pcu.signed_distance_to_mesh(query_points_np, vertices_np_f64, faces_np)

    # Convert back to tensor
    sdf_gt = torch.from_numpy(sdf_gt_np).float().unsqueeze(0).unsqueeze(-1).to(device) # [1, N_query, 1]
    return sdf_gt



@torch.no_grad() # Disable gradient calculation for inference
def extract_mesh(model, points_surface, bounds, config, device):
    """Extracts mesh using marching cubes."""
    extract_cfg = config["mesh_extraction"]
    model.eval() # Set model to evaluation mode

    print("\n--- Starting Mesh Extraction ---")
    print(f"Grid resolution: {extract_cfg['resolution']}x{extract_cfg['resolution']}x{extract_cfg['resolution']}")

    # Create grid points
    res = extract_cfg["resolution"]
    padding = extract_cfg["padding"]
    bounds_min = bounds[0] - padding
    bounds_max = bounds[1] + padding
    grid_min = torch.tensor(bounds_min, dtype=torch.float32, device=device)
    grid_max = torch.tensor(bounds_max, dtype=torch.float32, device=device)

    x = torch.linspace(grid_min[0], grid_max[0], res, device=device)
    y = torch.linspace(grid_min[1], grid_max[1], res, device=device)
    z = torch.linspace(grid_min[2], grid_max[2], res, device=device)
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij') # Note: indexing='ij' is important!
    grid_points = torch.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], dim=-1) # Shape: [res*res*res, 3]

    # Query SDF on the grid in batches
    sdf_values = []
    num_grid_points = grid_points.shape[0]
    batch_points = extract_cfg["batch_points"]

    print(f"Querying {num_grid_points} grid points for SDF...")
    for i in tqdm(range(0, num_grid_points, batch_points)):
        batch = grid_points[i : i + batch_points].unsqueeze(0) # Add batch dim: [1, N_batch, 3]
        # Ensure points_surface has batch dim [1, N_input, 3] if needed by model forward
        sdf_batch = model(points_surface, batch) # Shape: [1, N_batch, 1]
        sdf_values.append(sdf_batch.squeeze(0).cpu()) # Remove batch dim, move to CPU

    sdf_grid = torch.cat(sdf_values, dim=0).numpy().reshape(res, res, res) # Shape: [res, res, res]

    # Run Marching Cubes
    print("Running Marching Cubes...")
    try:
        # The spacing argument is crucial for getting the mesh in the correct coordinates
        spacing = ((grid_max - grid_min).cpu().numpy()) / (res - 1)
        vertices, faces, normals, _ = marching_cubes(
            volume=sdf_grid,
            level=0.0, # Extract the zero-level set
            spacing=spacing # Use grid spacing
        )
        # Adjust vertices origin to match the grid_min
        vertices += grid_min.cpu().numpy()

        print(f"Marching Cubes generated mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces.")

        # Create Trimesh object
        extracted_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

        # Save the extracted mesh
        extracted_mesh.export(extract_cfg["output_path"])
        print(f"--- Mesh Extraction Finished ---")
        print(f"Extracted mesh saved to: {extract_cfg['output_path']}")
        return extracted_mesh

    except ValueError as e:
        print(f"Marching Cubes Error: {e}. This might happen if the SDF grid doesn't cross the level set (0.0).")
        print("Try adjusting grid bounds/padding, resolution, or check model training.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during Marching Cubes: {e}")
        return None