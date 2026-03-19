import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os

# --- PyTorch3D Imports (Corrected) ---
try:
    from pytorch3d.ops import (
        sample_farthest_points,
        knn_points,           # Finds neighbors indices and distances
        knn_gather            # Gathers features using indices
    )
    # Removed three_nn, three_interpolate, feature_interpolate
    PYTORCH3D_AVAILABLE = True
    print("PyTorch3D found.")
except ImportError as e:
    print(f"PyTorch3D import error: {e}")
    print("Cannot use PyTorch3D optimizations. Ensure PyTorch3D is installed correctly.")
    print("See: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md")
    PYTORCH3D_AVAILABLE = False
    # Fallback knn definition (if needed)
    def knn(x, k):
        with torch.no_grad():
            inner = -2 * torch.matmul(x.transpose(2, 1), x)
            xx = torch.sum(x**2, dim=1, keepdim=True)
            pairwise_distance = -xx - inner - xx.transpose(2, 1)
            k = min(k, x.size(-1))
            if k <= 0: return torch.zeros(*pairwise_distance.shape[:-1], 0, dtype=torch.long, device=x.device)
            _, idx = pairwise_distance.topk(k=k, dim=-1)
        return idx

if not PYTORCH3D_AVAILABLE:
     raise ImportError("This script requires PyTorch3D for optimized operations, but it was not found or failed to import.")

# # --- Configuration ---
# # (Identical to previous version)
# N_EPOCHS = 5000
# INITIAL_LEARNING_RATE = 0.001
# WEIGHT_DECAY = 1e-6
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# PRINT_INTERVAL = 200
# DATA_FILE = 'example-plc_norm.xyz'
# COSINE_LOSS_WEIGHT = 1.0
# L2_LOSS_WEIGHT = 0.1

# # --- 1. Data Loading ---
# # (Identical to previous version)
# print("--- Loading Data ---")
# if not os.path.exists(DATA_FILE):
#     raise FileNotFoundError(f"Data file '{DATA_FILE}' not found.")
# point_and_normals = np.loadtxt(DATA_FILE)
# points_np = point_and_normals[:, :3]
# normals_np = point_and_normals[:, 3:]
# normals_norm = np.linalg.norm(normals_np, axis=1, keepdims=True)
# zero_norm_mask = (normals_norm < 1e-8)
# normals_norm[zero_norm_mask] = 1.0
# normals_np = normals_np / normals_norm
# normals_np[zero_norm_mask.squeeze()] = np.array([1.0, 0.0, 0.0])
# N_POINTS_LOADED = points_np.shape[0]
# print(f"Loaded {N_POINTS_LOADED} points and normals from {DATA_FILE}.")
# points_tensor = torch.from_numpy(points_np).float().unsqueeze(0).to(DEVICE)
# normals_tensor = torch.from_numpy(normals_np).float().unsqueeze(0).to(DEVICE)
# print(f"Using device: {DEVICE}")
# print(f"Points tensor shape: {points_tensor.shape}")
# print(f"Normals tensor shape: {normals_tensor.shape}")
# print("-" * 30)


# --- 2. PointNet++ Model Components (Using PyTorch3D) ---

# --- PointNetSetAbstractionPytorch3D ---
# (Identical to previous working version - uses sample_farthest_points, knn_points, knn_gather)
class PointNetSetAbstractionPytorch3D(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstractionPytorch3D, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        mlp_in_channel = 3 + in_channel
        last_channel = mlp_in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        B, C_xyz, N = xyz.shape
        C_feature = points.shape[1] if points is not None else 0
        last_mlp_channel = self.mlp_convs[-1].out_channels
        xyz_trans = xyz.permute(0, 2, 1).contiguous()
        points_trans = points.permute(0, 2, 1).contiguous() if points is not None else None

        if self.group_all:
            if N == 0: return xyz.new_zeros((B, C_xyz, 0)), xyz.new_zeros((B, last_mlp_channel, 0))
            new_xyz_trans = xyz_trans.mean(dim=1, keepdim=True)
            new_xyz = new_xyz_trans.permute(0, 2, 1)
            grouped_xyz_rel = (xyz - new_xyz).unsqueeze(2)
            if points is not None:
                 grouped_points_feat = points.unsqueeze(2)
                 mlp_input = torch.cat([grouped_xyz_rel, grouped_points_feat], dim=1)
            else:
                 mlp_input = grouped_xyz_rel
        else:
            S = self.npoint
            if S is None or S >= N:
                S = N
                sampled_indices = torch.arange(N, device=xyz.device).unsqueeze(0).expand(B, N)
                new_xyz_trans = xyz_trans
            elif S <= 0 :
                return xyz.new_zeros((B, C_xyz, 0)), xyz.new_zeros((B, last_mlp_channel, 0))
            else:
                _, sampled_indices = sample_farthest_points(xyz_trans, K=S)
                new_xyz_trans = torch.gather(xyz_trans, 1, sampled_indices.unsqueeze(-1).expand(-1, -1, 3))

            new_xyz = new_xyz_trans.permute(0, 2, 1)
            K = min(self.nsample, N)
            if K <= 0:
                 return new_xyz, xyz.new_zeros((B, last_mlp_channel, S))

            knn_result = knn_points(new_xyz_trans, xyz_trans, K=K, return_nn=True)
            knn_idx = knn_result.idx
            grouped_xyz_trans = knn_result.knn
            if points_trans is not None:
                grouped_points_trans = knn_gather(points_trans, knn_idx)
            else:
                grouped_points_trans = None

            grouped_xyz = grouped_xyz_trans.permute(0, 3, 1, 2).contiguous()
            grouped_xyz_rel = grouped_xyz - new_xyz.unsqueeze(-1)
            if grouped_points_trans is not None:
                grouped_points_feat = grouped_points_trans.permute(0, 3, 1, 2).contiguous()
                mlp_input = torch.cat([grouped_xyz_rel, grouped_points_feat], dim=1)
            else:
                mlp_input = grouped_xyz_rel

        new_points_processed = mlp_input
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points_processed = F.relu(bn(conv(new_points_processed)))
        new_points_pooled = torch.max(new_points_processed, dim=3)[0]
        return new_xyz, new_points_pooled


# --- PointNetFeaturePropagationPytorch3D (Corrected Interpolation Logic) ---
class PointNetFeaturePropagationPytorch3D(nn.Module):
    """Feature Propagation Layer using PyTorch3D k-NN and manual interpolation."""
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagationPytorch3D, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        B, C, N = xyz1.shape
        _, _, S = xyz2.shape
        D1 = points1.shape[1] if points1 is not None else 0
        D2 = points2.shape[1] if points2 is not None else 0
        last_mlp_channel = self.mlp_convs[-1].out_channels

        if N == 0: return xyz1.new_zeros((B, last_mlp_channel, 0))

        xyz1_trans = xyz1.permute(0, 2, 1).contiguous()
        xyz2_trans = xyz2.permute(0, 2, 1).contiguous()
        points1_trans = points1.permute(0, 2, 1).contiguous() if points1 is not None else None
        points2_trans = points2.permute(0, 2, 1).contiguous() if points2 is not None else None

        if points2_trans is None or D2 == 0 or S == 0:
            interpolated_points = xyz1.new_zeros((B, D2, N))
        elif S == 1:
             interpolated_points = points2.repeat(1, 1, N)
        else:
            # --- Interpolation using knn_points, manual weights, knn_gather ---
            K_INTERP = 3 # Number of neighbors for interpolation
            knn_result = knn_points(xyz1_trans, xyz2_trans, K=K_INTERP) # Find K nearest neighbors
            idx = knn_result.idx     # (B, N, K_INTERP) Indices of neighbors in xyz2_trans
            dists = knn_result.dists # (B, N, K_INTERP) Squared distances

            # Inverse distance weighting
            dist_recip = 1.0 / torch.clamp(dists, min=1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / torch.clamp(norm, min=1e-8) # (B, N, K_INTERP) Normalized weights

            # Gather features of the K neighbors
            gathered_features = knn_gather(points2_trans, idx) # (B, N, K_INTERP, D2)

            # Perform weighted sum: (B, N, K_INTERP, D2) * (B, N, K_INTERP, 1) -> sum over K_INTERP dim
            interpolated_points_trans = torch.sum(gathered_features * weight.unsqueeze(-1), dim=2) # (B, N, D2)

            # Transpose back to (B, D2, N)
            interpolated_points = interpolated_points_trans.permute(0, 2, 1).contiguous()

        # --- Concatenate Skip Features ---
        if points1 is not None:
            if points1.shape[2] != N: raise ValueError(f"Skip shape mismatch: p1 {points1.shape}, N {N}")
            if interpolated_points.dim() == 2 and D2 == 0: interpolated_points = interpolated_points.unsqueeze(1)
            if interpolated_points.shape[2] != N: raise ValueError(f"Interp shape mismatch: interp {interpolated_points.shape}, N {N}")
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        if new_points.shape[1] == 0:
             return xyz1.new_zeros((B, last_mlp_channel, N))

        # --- Apply MLPs ---
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

# --- PointNet2NormalPytorch3D Model Definition ---
# (Uses the corrected FP layer)
class PointNet2NormalPytorch3D(nn.Module):
    def __init__(self, num_classes=3, use_xyz_features=True):
        super(PointNet2NormalPytorch3D, self).__init__()
        self.use_xyz = use_xyz_features
        l0_feat_dim = 3 if self.use_xyz else 0
        # Hyperparameters remain the same as the aggressive overfitting version
        sa1_npoint = 512; sa1_nsample = 96; sa1_mlp = [128, 128, 256]; sa1_feat_out = sa1_mlp[-1]
        sa2_npoint = 128; sa2_nsample = 128; sa2_mlp = [256, 256, 512]; sa2_feat_out = sa2_mlp[-1]
        sa3_npoint = 32;  sa3_nsample = 192; sa3_mlp = [512, 512, 1024]; sa3_feat_out = sa3_mlp[-1]
        fp3_feat_in = sa2_feat_out + sa3_feat_out; fp3_mlp = [1024, 512]; fp3_feat_out = fp3_mlp[-1]
        fp2_feat_in = sa1_feat_out + fp3_feat_out; fp2_mlp = [512, 256]; fp2_feat_out = fp2_mlp[-1]
        fp1_feat_in = l0_feat_dim + fp2_feat_out; fp1_mlp = [256, 128]; fp1_feat_out = fp1_mlp[-1]
        final_head_in = fp1_feat_out
        # Instantiating the layers
        self.sa1 = PointNetSetAbstractionPytorch3D(npoint=sa1_npoint, radius=0.1, nsample=sa1_nsample, in_channel=l0_feat_dim, mlp=sa1_mlp)
        self.sa2 = PointNetSetAbstractionPytorch3D(npoint=sa2_npoint, radius=0.2, nsample=sa2_nsample, in_channel=sa1_feat_out, mlp=sa2_mlp)
        self.sa3 = PointNetSetAbstractionPytorch3D(npoint=sa3_npoint, radius=0.4, nsample=sa3_nsample, in_channel=sa2_feat_out, mlp=sa3_mlp)
        self.fp3 = PointNetFeaturePropagationPytorch3D(in_channel=fp3_feat_in, mlp=fp3_mlp)
        self.fp2 = PointNetFeaturePropagationPytorch3D(in_channel=fp2_feat_in, mlp=fp2_mlp)
        self.fp1 = PointNetFeaturePropagationPytorch3D(in_channel=fp1_feat_in, mlp=fp1_mlp)
        self.conv1 = nn.Conv1d(final_head_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz_input):
        # Forward pass logic remains the same
        B, N, C = xyz_input.shape
        xyz = xyz_input.permute(0, 2, 1)
        l0_xyz = xyz
        l0_points = l0_xyz if self.use_xyz else None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l2_points_fp = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points_fp = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_fp)
        l0_points_fp = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points_fp)
        x = F.relu(self.bn1(self.conv1(l0_points_fp)))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = F.normalize(x, p=2, dim=2, eps=1e-8)
        return x

# --- 3. Loss Function ---
# (Combined Loss - Identical)
def combined_loss(pred_normals, true_normals, cos_weight=1.0, l2_weight=0.1):
    pred_norm = F.normalize(pred_normals, p=2, dim=2, eps=1e-8)
    true_norm = F.normalize(true_normals, p=2, dim=2, eps=1e-8)
    cos_sim = torch.sum(pred_norm * true_norm, dim=2)
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
    cosine_loss = 1.0 - cos_sim
    l2_loss = torch.sum((pred_norm - true_norm)**2, dim=2)
    mean_cosine_loss = torch.mean(cosine_loss)
    mean_l2_loss = torch.mean(l2_loss)
    total_loss = (cos_weight * mean_cosine_loss) + (l2_weight * mean_l2_loss)
    return total_loss



def cos_angle(v1, v2):
    """
        V1, V2: (N, 3)
        return: (N,)
    """
    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)

import pymeshlab
from src import RIMLSProcessor
class NetworkMLS(nn.Module):
    def __init__(self, num_points, num_knn, points):
        super(NetworkMLS, self).__init__()
        self.num_points = num_points
        self.num_knn = num_knn
        self.num_iter = 2
        self.points = points.float().cuda()
        self.processor = RIMLSProcessor()
        self.processor.build_index(points)
        
        # Initialize normals as a parameter directly
        normals_init = torch.randn_like(self.points)
        # Fix normal normalization - use dim=1 instead of axis=0 to normalize along the feature dimension
        normals_init = normals_init / torch.linalg.norm(normals_init, dim=1, keepdim=True)
        m = pymeshlab.Mesh(self.points.detach().cpu().numpy())
        ms = pymeshlab.MeshSet()
        ms.add_mesh(m)
        ms.compute_normal_for_point_clouds(k=10)
        result_mesh = ms.current_mesh()
        normals_init = result_mesh.vertex_normal_matrix()
        normals_init = torch.from_numpy(normals_init).float().cuda()
        self.normals = nn.Parameter(normals_init)
    def sdf(self, x):
        # Normalize normals without reassigning the parameter
        normalized_normals = F.normalize(self.normals, p=2, dim=1, eps=1e-8)
        # print(self.points.shape)
        sdf, grad = self.processor(x.squeeze(0), self.points, normalized_normals.squeeze(0), compute_gradient = True)
        return sdf.unsqueeze(0).unsqueeze(2), grad.unsqueeze(0)
    def forward(self, pcl_source):
        """
            pcl_source: (*, N, 3)
        """
        self.sd_all = []
        self.grad_all = []
        with torch.set_grad_enabled(True):
            pcl_source.requires_grad = True
            sd_temp = torch.zeros_like(pcl_source)[::,0:1]
            grad_temp = torch.zeros_like(pcl_source)
        
            for i in range(self.num_iter):
                pcl_source = pcl_source - sd_temp * grad_temp

                sd_temp, grad_temp = self.sdf(pcl_source)     # (*, N, 1), (*, N, 3)
                # print(sd_temp.shape)
                # print(grad_temp.shape)
                self.sd_all.append(sd_temp)
                self.grad_all.append(grad_temp)

                if i == 0:
                    self.sd = sd_temp
                    self.grad_norm = grad_temp
                elif i == 1:
                    self.sd1 = sd_temp
                    self.grad_norm1 = grad_temp
                elif i == 2:
                    self.sd2 = sd_temp
                    self.grad_norm2 = grad_temp
                else:
                    raise ValueError('Not set value')

            self.grad_sum = F.normalize(sum(self.grad_all), dim=-1)

        return self.grad_sum

    def get_loss(self, pcl_raw=None, pcl_source=None, knn_idx=None):
        """
            pcl_raw: (1, M, 3), M >= N
            pcl_source: (1, N+n, 3)
            normal_gt: (1, N, 3)
            knn_idx: (1, N, K)
        """
        num_points = self.num_points
        _device, _dtype = pcl_source.device, pcl_source.dtype
        loss_d = torch.zeros(1, device=_device, dtype=_dtype)
        loss_v1 = torch.zeros(1, device=_device, dtype=_dtype)
        loss_v2 = torch.zeros(1, device=_device, dtype=_dtype)
        loss_v3 = torch.zeros(1, device=_device, dtype=_dtype)
        loss_reg1 = torch.zeros(1, device=_device, dtype=_dtype)
        loss_reg2 = torch.zeros(1, device=_device, dtype=_dtype)
        loss_con = torch.zeros(1, device=_device, dtype=_dtype)
        loss_sd = torch.zeros(1, device=_device, dtype=_dtype)

        pcl_nn = knn_gather(pcl_raw, knn_idx)                   # (1, N, K, 3)
        v = pcl_source[:, :num_points, None, :3] - pcl_nn       # (1, N, K, 3)
        v1 = v[:,:,:8,:].mean(-2)                               # (1, N, 3)
        v2 = v[:,:,:4,:].mean(-2)                               # (1, N, 3)
        v3 = v[:,:,0,:]                                         # (1, N, 3)

        pcl_target = torch.cat((pcl_nn[:,:,0,:], pcl_source[:, num_points:, :]), dim=-2)

        loss_reg1 = 10 * (self.sd[:, num_points:, :]**2).mean()
        loss_reg2 = 10 * (self.sd1**2).mean() #+ 10 * (self.sd2**2).mean()

        weight = torch.exp(-60 * torch.abs(self.sd)).squeeze()      # (N,)

        loss_v1 = torch.linalg.norm((v1 - (self.sd * self.grad_norm)[:, :num_points, :]), ord=2, dim=-1).mean()
        loss_v2 = torch.linalg.norm((v2 - (self.sd * self.grad_norm)[:, :num_points, :]), ord=2, dim=-1).mean()
        loss_v3 = torch.linalg.norm((v3 - (self.sd * self.grad_norm)[:, :num_points, :]), ord=2, dim=-1).mean()

        pcl_source_new = pcl_source - self.sd * self.grad_norm - self.sd1 * self.grad_norm1 #- self.sd2 * self.grad_norm2
        loss_d = 0.3 * torch.linalg.norm((pcl_source_new - pcl_target), ord=2, dim=-1).mean()

        cos_ang = cos_angle(self.grad_norm[0, :, :], self.grad_norm1[0, :, :])  # (N,)
        # cos_ang1 = cos_angle(self.grad_norm[0, :, :], self.grad_norm2[0, :, :])
        loss_con = 0.01 * (weight * (1 - cos_ang)).mean() #+ 0.01 * (weight * (1 - cos_ang1)).mean()

        # loss_sd = 0.01 * torch.clamp(torch.abs(self.sd + self.sd1)[:, :num_points, :] - torch.linalg.norm(v3, ord=2, dim=-1), min=0.0).mean()

        loss_tuple = (loss_v1, loss_v2, loss_v3, loss_d, loss_reg1, loss_reg2, loss_con, loss_sd)
        loss_sum = sum(loss_tuple)
        return loss_sum, loss_tuple