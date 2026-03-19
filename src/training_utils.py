import torch 
from tqdm import tqdm
import numpy as np
from skimage import measure
import trimesh

def save_normals_xyz(model, pointcloud, save_dir, filename = 'estimated_normals', batch_size = 1000, verbose = False, cal_attention_weights = False):
    '''
    Utility function used to save the pointcloud and normals to disk
    in xyz format. 
    '''
    #### [(1)] Initialize the query and full pointclouds.

    # The model uses autograd to compute the graident / normal field.
    query_points    = pointcloud.clone().requires_grad_(True)
    full_pointcloud = pointcloud.clone().requires_grad_(True)
    
    full_pointcloud_np = full_pointcloud.detach().cpu().numpy()

    num_batches = (query_points.shape[0] + batch_size - 1) // batch_size
    
    # Initialize tensors to store results
    all_sdf  = torch.zeros((full_pointcloud.shape[0], 1), device=full_pointcloud.device)
    all_grad = torch.zeros((full_pointcloud.shape[0], 3), device=full_pointcloud.device)
    attention_weights = torch.zeros((full_pointcloud.shape[0], model.num_anchor_points), device=full_pointcloud.device)

    #### [(2)] Compute the SDF and gradient for each batch.
    if verbose:
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, full_pointcloud.shape[0])
            
            batch_sdf, batch_grad = model.gradient(
                full_pointcloud.unsqueeze(0),
                query_points[start_idx:end_idx].unsqueeze(0)
            )
            if cal_attention_weights:
                batch_attn_weights = model.cross_attn_weights_inference
                attention_weights[start_idx:end_idx] = batch_attn_weights.detach().squeeze()
            # Important: detach() is used to prevent the gradients from being computed
            #            on the full pointcloud.
            all_sdf[start_idx:end_idx]  = batch_sdf.detach()
            all_grad[start_idx:end_idx] = batch_grad.detach()
    else:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, full_pointcloud.shape[0])
            
            batch_sdf, batch_grad = model.gradient(
                full_pointcloud.unsqueeze(0),
                query_points[start_idx:end_idx].unsqueeze(0)
            )
            if cal_attention_weights:
                batch_attn_weights = model.cross_attn_weights_inference
                attention_weights[start_idx:end_idx] = batch_attn_weights.detach().squeeze()
            all_sdf[start_idx:end_idx]  = batch_sdf.detach()
            all_grad[start_idx:end_idx] = batch_grad.detach()

    #### [(3)] Save the results to disk.
    grad_norm = all_grad.cpu().detach().numpy()
    pred_norm = grad_norm.copy()  # Use copy() for numpy arrays
    pred_norm[np.linalg.norm(pred_norm, axis=-1) == 0.0] = 1.0
    pred_norm = pred_norm / np.linalg.norm(pred_norm, axis=-1, keepdims=True)  # Use non-inplace division

    
    pred_norm  = np.squeeze(pred_norm)
    output_xyz = np.hstack([full_pointcloud_np, pred_norm])
    np.savetxt(f'{save_dir}/{filename}.xyz', output_xyz, fmt='%.6f %.6f %.6f %.6f %.6f %.6f')
    np.save(f'{save_dir}/attention_weights_points.npy', attention_weights.detach().cpu().numpy())
    return torch.from_numpy(pred_norm).float().to(full_pointcloud.device)



def run_rimls(processor,points, normals, resolution = 100, batch_size = 10000, verbose = False):

    points  = points
    normals = normals#.detach().cpu().numpy()


    #### [(1)] Create the meshgrid.
    min_bounds = points.min(dim=0)[0]
    max_bounds = points.max(dim=0)[0]
    x = torch.linspace(min_bounds[0].item(), max_bounds[0].item(), resolution)
    y = torch.linspace(min_bounds[1].item(), max_bounds[1].item(), resolution)
    z = torch.linspace(min_bounds[2].item(), max_bounds[2].item(), resolution)

    # Create the meshgrid
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1).cuda()

    ### [(2)] Evaluate the RIMLS field at the grid points.
    potential = torch.zeros(grid_points.shape[0], device=grid_points.device)
    gradient  = torch.zeros(grid_points.shape[0], 3, device=grid_points.device)

    num_batches = (grid_points.shape[0] + batch_size - 1) // batch_size
    if verbose:
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, grid_points.shape[0])

        batch_potential, batch_gradient = processor.forward(grid_points[start_idx:end_idx],
                                                            points.detach(), 
                                                            normals,
                                                            compute_gradient=True)
        potential[start_idx:end_idx] = batch_potential.detach().squeeze()
        gradient[start_idx:end_idx]  = batch_gradient.detach().squeeze()
    else:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, grid_points.shape[0])
            batch_potential, batch_gradient = processor.forward(grid_points[start_idx:end_idx],
                                                                points.detach(), 
                                                                normals,
                                                                compute_gradient=True)
            potential[start_idx:end_idx] = batch_potential.detach().squeeze()
            gradient[start_idx:end_idx]  = batch_gradient.detach().squeeze()
    return potential.reshape(resolution, resolution, resolution), gradient.reshape(resolution, resolution, resolution, 3)


def run_marching_cubes(sdf_values_grid, points, normals, save_dir, filename = 'output_mesh', iso_value = 0.0, resolution = 100):
    
    sdf_values_grid = sdf_values_grid.detach().cpu().numpy()
    
    np.save(f'{save_dir}/{filename}_sdf_values_grid.npy', sdf_values_grid)
    points  = points #.detach().cpu().numpy()
    normals = normals#.detach().cpu().numpy()

    #### [(1)] Create the meshgrid.
    min_bounds = points.min(dim=0)[0]
    max_bounds = points.max(dim=0)[0]

    volume_size = max_bounds - min_bounds
    spacing = volume_size / (resolution - 1)
    spacing = spacing.detach().cpu().numpy()
    vertices, faces, normals, values = measure.marching_cubes(sdf_values_grid, iso_value, spacing=tuple(spacing))

    vertices_world = vertices + min_bounds.detach().cpu().numpy()
    mesh = trimesh.Trimesh(vertices=vertices_world, faces=faces)
    mesh.vertex_normals

    # Save the mesh to disk
    output_path = f"{save_dir}/{filename}.ply"
    mesh.export(output_path)
    return mesh


def run_field_cross_attn(model, points, batch_size = 10000, verbose = False, cal_attention_weights = False):
    min_bounds = points.min(dim=0)[0]
    max_bounds = points.max(dim=0)[0]
    resolution = 100
    x = torch.linspace(min_bounds[0].item(), max_bounds[0].item(), resolution)
    y = torch.linspace(min_bounds[1].item(), max_bounds[1].item(), resolution)
    z = torch.linspace(min_bounds[2].item(), max_bounds[2].item(), resolution)

    # Create the meshgrid
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1).cuda()

    ### [(2)] Evaluate the RIMLS field at the grid points.
    potential = torch.zeros(grid_points.shape[0], device=grid_points.device)
    gradient  = torch.zeros(grid_points.shape[0], 3, device=grid_points.device)
    attention_weights = torch.zeros(grid_points.shape[0], model.num_anchor_points, device=grid_points.device)
    num_batches = (grid_points.shape[0] + batch_size - 1) // batch_size
    if verbose:
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, grid_points.shape[0])

        batch_potential = model.forward(points, grid_points[start_idx:end_idx].unsqueeze(0),)
        if cal_attention_weights:
            batch_attn_weights = model.cross_attn_weights_inference
            attention_weights[start_idx:end_idx] = batch_attn_weights.detach().squeeze()
        potential[start_idx:end_idx] = batch_potential.detach().squeeze()
        # gradient[start_idx:end_idx]  = batch_gradient.detach().squeeze()
    else:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, grid_points.shape[0])
            batch_potential = model.forward(points, grid_points[start_idx:end_idx].unsqueeze(0))
            potential[start_idx:end_idx] = batch_potential.detach().squeeze()
            if cal_attention_weights:
                batch_attn_weights = model.cross_attn_weights_inference
                attention_weights[start_idx:end_idx] = batch_attn_weights.detach().squeeze()
            # gradient[start_idx:end_idx]  = batch_gradient.detach().squeeze()
    if cal_attention_weights:
        np.save(f'attention_weights.npy', attention_weights.detach().cpu().numpy())
        np.save(f'attention_weights_sdf_field.npy', potential.reshape(resolution, resolution, resolution).detach().cpu().numpy())
    return potential.reshape(resolution, resolution, resolution), None