import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_gather
import numpy as np


def cos_angle(v1, v2):
    """
        V1, V2: (N, 3)
        return: (N,)
    """
    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)


class MLPNet_linear(nn.Module):
    def __init__(self,
                 d_in=3,
                 d_mid=256,
                 d_out=1,
                 n_mid=8,
                 bias=0.5,
                 geometric_init=True,
                 weight_norm=False,
                 inside_grad=True,
            ):
        super(MLPNet_linear, self).__init__()
        assert n_mid > 3
        dims = [d_in] + [d_mid for _ in range(n_mid)] + [d_out]
        self.num_layers = len(dims)
        self.skip_in = [n_mid // 2]

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - d_in
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)
            if geometric_init:
                if l == self.num_layers - 2:
                    if inside_grad:  # inside SDF > 0
                        nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        nn.init.constant_(lin.bias, bias)
                    else:
                        nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        nn.init.constant_(lin.bias, -bias)
                else:
                    nn.init.normal_(lin.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(out_dim))
                    nn.init.constant_(lin.bias, 0.0)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

    def forward(self, pos):
        """
            pos: (*, N, C)
        """
        x = pos
        for l in range(0, self.num_layers - 1):
            if l in self.skip_in:
                x = torch.cat([x, pos], dim=-1)
                x = x / np.sqrt(2)

            lin = getattr(self, "lin" + str(l))
            x = lin(x)

            if l < self.num_layers - 2:
                x = F.relu(x)
        return x

    def gradient(self, x):
        """
            x: (*, N, C), with requires_grad is set to true
        """
        y = self.forward(x)             # (*, N, 1), signed distance

        # y.sum().backward(retain_graph=True)
        # grad_out = x.grad.detach()

        grad_outputs = torch.ones_like(y, requires_grad=False, device=y.device)
        grad_out = torch.autograd.grad(outputs=y,
                                    inputs=x,
                                    grad_outputs=grad_outputs,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
        grad_norm = F.normalize(grad_out, dim=-1)     # (*, N, 3)
        return y, grad_norm


class Network(nn.Module):
    def __init__(self, num_points, num_knn):
        super(Network, self).__init__()
        self.num_points = num_points
        self.num_knn = num_knn
        self.num_iter = 2

        self.net = MLPNet_linear(d_in=3, d_mid=256, d_out=1, n_mid=8)

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

                sd_temp, grad_temp = self.net.gradient(pcl_source)     # (*, N, 1), (*, N, 3)
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


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# # --- Helper Modules ---

# class PositionalEncoder(nn.Module):
#     """Sine-cosine positional encoder for input points."""
#     def __init__(self, d_input=3, n_freqs=10, log_space=True):
#         super().__init__()
#         self.d_input = d_input
#         self.n_freqs = n_freqs
#         self.log_space = log_space
#         self.d_output = d_input * (1 + 2 * self.n_freqs)
#         self.embed_fns = [lambda x: x]

#         # Define frequencies
#         if self.log_space:
#             freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
#         else:
#             freq_bands = torch.linspace(1., 2.**(self.n_freqs - 1), self.n_freqs)

#         for freq in freq_bands:
#             self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
#             self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

#     def forward(self, coords):
#         """
#         Apply positional encoding to coordinates.
#         Args:
#             coords (torch.Tensor): Shape (B, ..., d_input)
#         Returns:
#             torch.Tensor: Shape (B, ..., d_output)
#         """
#         return torch.cat([fn(coords) for fn in self.embed_fns], dim=-1)

# class MLP(nn.Module):
#     """A simple MLP module."""
#     def __init__(self, d_in, d_out, d_hidden, n_layers, activation=nn.ReLU, use_layernorm=False):
#         super().__init__()
#         layers = []
#         in_dim = d_in
#         for i in range(n_layers):
#             layers.append(nn.Linear(in_dim, d_hidden))
#             if use_layernorm:
#                 layers.append(nn.LayerNorm(d_hidden))
#             layers.append(activation())
#             in_dim = d_hidden
#         layers.append(nn.Linear(d_hidden, d_out))
#         self.mlp = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.mlp(x)

# class MultiHeadAttention(nn.Module):
#     """ Standard Multi-Head Attention """
#     def __init__(self, d_model, n_heads, dropout=0.1):
#         super().__init__()
#         assert d_model % n_heads == 0
#         self.d_k = d_model // n_heads
#         self.n_heads = n_heads
#         self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)]) # Wq, Wk, Wv, Wo
#         self.dropout = nn.Dropout(p=dropout)
#         self.scale = 1. / math.sqrt(self.d_k)

#     def forward(self, query, key, value, mask=None):
#         # query: (B, N_q, d_model), key: (B, N_k, d_model), value: (B, N_v, d_model) N_k==N_v
#         batch_size = query.size(0)

#         # 1) Linear projections & split into heads: (B, N, d_model) -> (B, N, H, d_k) -> (B, H, N, d_k)
#         query, key, value = [
#             l(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
#             for l, x in zip(self.linears, (query, key, value))
#         ]

#         # 2) Apply attention on all the projected vectors in batch.
#         # scores: (B, H, N_q, N_k)
#         scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

#         if mask is not None:
#              # Ensure mask has compatible shape: e.g., (B, 1, 1, N_k) or (B, 1, N_q, N_k)
#             scores = scores.masked_fill(mask == 0, -1e9)

#         # attn_weights: (B, H, N_q, N_k)
#         attn_weights = F.softmax(scores, dim=-1)
#         attn_weights = self.dropout(attn_weights)

#         # context: (B, H, N_q, d_k)
#         context = torch.matmul(attn_weights, value)

#         # 3) "Concat" using view and apply final linear.
#         # context: (B, N_q, d_model)
#         context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
#         return self.linears[-1](context) # Apply Wo

# class AttentionLayer(nn.Module):
#     """ Generic attention layer with residual connection and layer norm """
#     def __init__(self, d_model, n_heads, dropout=0.1, use_residual=True, use_layernorm=True):
#         super().__init__()
#         self.attention = MultiHeadAttention(d_model, n_heads, dropout)
#         self.use_residual = use_residual
#         self.use_layernorm = use_layernorm
#         if self.use_layernorm:
#             self.norm = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(p=dropout)

#     def forward(self, query, key, value, mask=None):
#         # Allow for self-attention (q=k=v) or cross-attention
#         context = self.attention(query, key, value, mask)
#         context = self.dropout(context)

#         if self.use_residual:
#              # Residual connection only makes sense if query dim == context dim
#             if query.shape == context.shape:
#                 output = query + context
#             else: # Fallback if dims dont match (e.g., first cross-attention)
#                 output = context
#         else:
#             output = context

#         if self.use_layernorm:
#             output = self.norm(output)

#         return output

# # --- K-Nearest Neighbors (Simple Implementation) ---
# # Note: For larger point clouds, use libraries like PyTorch Geometric
# # or FAISS for efficient KNN.

# def find_knn(query_points, full_pc, k):
#     """
#     Find k-nearest neighbors in full_pc for each query point.
#     Args:
#         query_points (torch.Tensor): Query points (B, 3)
#         full_pc (torch.Tensor): Full point cloud (N, 3)
#         k (int): Number of neighbors
#     Returns:
#         neighbor_indices (torch.Tensor): Indices of neighbors (B, k)
#         neighbor_points (torch.Tensor): Coordinates of neighbors (B, k, 3)
#     """
#     with torch.no_grad(): # No need to track gradients for KNN search
#         B = query_points.shape[0]
#         N = full_pc.shape[0]

#         # Expand dims for broadcasting distance calculation
#         query_expanded = query_points.unsqueeze(1) # (B, 1, 3)
#         full_pc_expanded = full_pc.unsqueeze(0)   # (1, N, 3)

#         # Calculate squared Euclidean distances (more stable than sqrt)
#         dist_sq = torch.sum((query_expanded - full_pc_expanded)**2, dim=-1) # (B, N)

#         # Find the top k smallest distances and their indices
#         _, neighbor_indices = torch.topk(dist_sq, k, dim=-1, largest=False, sorted=True) # (B, k)

#         # Gather the actual neighbor coordinates
#         # Use gather or advanced indexing. Indexing is often clearer:
#         neighbor_points = full_pc[neighbor_indices] # (B, k, 3)

#     return neighbor_indices, neighbor_points


# # --- Main Network ---

# class AttentionNeuralGF(nn.Module):
#     def __init__(self,
#                  d_in=3,
#                  n_freqs=10,
#                  d_feature=256, # Feature dimension after initial MLP
#                  d_mlp_hidden=256,
#                  n_mlp_layers=4, # Layers in MLP_feat and MLP_final
#                  n_attn_heads=8,
#                  n_global_attn_layers=2,
#                  d_attn_hidden=256, # Dimension within attention feedforward
#                  dropout=0.1,
#                  k_neighbors=16,
#                  n_subsample_global=512):
#         super().__init__()
#         self.k_neighbors = k_neighbors
#         self.n_subsample_global = n_subsample_global
#         self.d_feature = d_feature

#         # 1. Positional Encoder
#         self.pos_encoder = PositionalEncoder(d_input=d_in, n_freqs=n_freqs)
#         d_pos_encoded = self.pos_encoder.d_output

#         # 2. Shared Feature Extractor Backbone
#         self.mlp_feat = MLP(d_pos_encoded, d_feature, d_mlp_hidden, n_mlp_layers, use_layernorm=True)

#         # 3. Global Context Module
#         self.global_self_attention = nn.ModuleList([
#             AttentionLayer(d_feature, n_attn_heads, dropout) for _ in range(n_global_attn_layers)
#         ])
#         self.global_cross_attention = AttentionLayer(d_feature, n_attn_heads, dropout)

#         # 4. Local Context Module
#         self.local_cross_attention = AttentionLayer(d_feature, n_attn_heads, dropout)

#         # 5. Fusion and Final Prediction MLP
#         # Input dimension is concatenation of q_feat, c_global, c_local
#         d_fusion = d_feature + d_feature + d_feature
#         self.mlp_final = MLP(d_fusion, 1, d_mlp_hidden, n_mlp_layers, use_layernorm=False) # Output is scalar distance

#         print(f"AttentionNeuralGF Initialized:")
#         print(f"  Positional Encoding Output Dim: {d_pos_encoded}")
#         print(f"  Shared Feature Dim (d_feature): {d_feature}")
#         print(f"  KNN Neighbors (k): {k_neighbors}")
#         print(f"  Global Subsample (M): {n_subsample_global}")
#         print(f"  Attention Heads: {n_attn_heads}")
#         print(f"  Final MLP Input Dim (d_fusion): {d_fusion}")

#     def forward(self, x_query, P_full):
#         """
#         Forward pass to predict signed distance.
#         Args:
#             x_query (torch.Tensor): Batch of query points (B, 3).
#             P_full (torch.Tensor): The *single* full point cloud for the current shape (N, 3).
#         Returns:
#             signed_distance (torch.Tensor): Predicted distance for each query point (B, 1).
#         """
#         B = x_query.shape[0]
#         N = P_full.shape[0]
#         device = x_query.device

#         # --- 1. Preprocessing ---
#         gamma_x = self.pos_encoder(x_query) # (B, d_pos_encoded)

#         # Subsample P_full for global context (do once if P_full doesn't change often within a batch)
#         # Ensure n_subsample_global is not larger than N
#         M = min(self.n_subsample_global, N)
#         if M < N :
#              # Random subsampling for simplicity
#             subset_indices = torch.randperm(N, device=device)[:M]
#         else:
#             subset_indices = torch.arange(N, device=device)

#         P_sub = P_full[subset_indices] # (M, 3)
#         gamma_P_sub = self.pos_encoder(P_sub) # (M, d_pos_encoded)

#         # Find KNN for local context
#         # neighbor_indices: (B, k), neighbor_points: (B, k, 3)
#         _, neighbor_points = find_knn(x_query, P_full, self.k_neighbors)
#         gamma_N_local = self.pos_encoder(neighbor_points) # (B, k, d_pos_encoded)

#         # --- 2. Shared Feature Extraction ---
#         q_feat = self.mlp_feat(gamma_x) # (B, d_feature)
#         F_sub = self.mlp_feat(gamma_P_sub) # (M, d_feature)
#         # Reshape for batch processing of MLP_feat on neighbors
#         gamma_N_local_flat = gamma_N_local.view(B * self.k_neighbors, -1)
#         F_local_flat = self.mlp_feat(gamma_N_local_flat)
#         F_local = F_local_flat.view(B, self.k_neighbors, self.d_feature) # (B, k, d_feature)

#         # --- 3. Global Context Module ---
#         # Add batch dim for attention layers (batch size = 1 for global context)
#         F_sub_batched = F_sub.unsqueeze(0) # (1, M, d_feature)
#         # 3a. Global Self-Attention
#         F_global_context = F_sub_batched
#         for attn_layer in self.global_self_attention:
#             F_global_context = attn_layer(F_global_context, F_global_context, F_global_context) # (1, M, d_feature)

#         # 3b. Global Cross-Attention (Query attends to global context)
#         # Query: (B, 1, d_feature), Key/Value: (1, M, d_feature) -> expand K/V to batch size B
#         q_feat_expanded = q_feat.unsqueeze(1) # (B, 1, d_feature)
#         F_global_context_expanded = F_global_context.expand(B, -1, -1) # (B, M, d_feature)
#         c_global = self.global_cross_attention(q_feat_expanded, F_global_context_expanded, F_global_context_expanded) # (B, 1, d_feature)
#         c_global = c_global.squeeze(1) # (B, d_feature)

#         # --- 4. Local Context Module ---
#         # Local Cross-Attention (Query attends to its k neighbors)
#         # Query: (B, 1, d_feature), Key/Value: (B, k, d_feature)
#         c_local = self.local_cross_attention(q_feat_expanded, F_local, F_local) # (B, 1, d_feature)
#         c_local = c_local.squeeze(1) # (B, d_feature)

#         # --- 5. Fusion and Final Prediction ---
#         f_fused = torch.cat([q_feat, c_global, c_local], dim=-1) # (B, d_fusion)
#         signed_distance = self.mlp_final(f_fused) # (B, 1)

#         return signed_distance

#     def get_gradient(self, x_query, P_full):
#         """
#         Compute the gradient of the signed distance w.r.t. input coordinates.
#         Args:
#             x_query (torch.Tensor): Batch of query points (B, 3).
#             P_full (torch.Tensor): The *single* full point cloud (N, 3).
#         Returns:
#             gradients (torch.Tensor): Normalized gradients (normals) (B, 3).
#         """
#         x_query.requires_grad_(True) # Ensure gradient tracking for query points

#         # Perform forward pass to get signed distance
#         s = self.forward(x_query, P_full) # (B, 1)

#         # Compute gradients
#         # Use grad_outputs=torch.ones_like(s) for scalar output -> vector gradient
#         gradients = torch.autograd.grad(
#             outputs=s,
#             inputs=x_query,
#             grad_outputs=torch.ones_like(s),
#             create_graph=True, # Keep graph for potential higher-order derivatives or training loop
#             retain_graph=True, # Keep graph if needed elsewhere
#             only_inputs=True
#         )[0] # Get gradient w.r.t. x_query

#         # Normalize gradients to get unit normals
#         # Add epsilon for numerical stability
#         normals = F.normalize(gradients, p=2, dim=-1, eps=1e-6)

#         # Detach gradient from query points if no longer needed upstream
#         # x_query.requires_grad_(False) # Or manage requires_grad outside

#         return normals
    
#     def get_loss(self, pcl_raw=None, pcl_source=None, knn_idx=None):
#         """
#             pcl_raw: (1, M, 3), M >= N
#             pcl_source: (1, N+n, 3)
#             normal_gt: (1, N, 3)
#             knn_idx: (1, N, K)
#         """
#         num_points = self.num_points
#         _device, _dtype = pcl_source.device, pcl_source.dtype
#         loss_d = torch.zeros(1, device=_device, dtype=_dtype)
#         loss_v1 = torch.zeros(1, device=_device, dtype=_dtype)
#         loss_v2 = torch.zeros(1, device=_device, dtype=_dtype)
#         loss_v3 = torch.zeros(1, device=_device, dtype=_dtype)
#         loss_reg1 = torch.zeros(1, device=_device, dtype=_dtype)
#         loss_reg2 = torch.zeros(1, device=_device, dtype=_dtype)
#         loss_con = torch.zeros(1, device=_device, dtype=_dtype)
#         loss_sd = torch.zeros(1, device=_device, dtype=_dtype)

#         pcl_nn = knn_gather(pcl_raw, knn_idx)                   # (1, N, K, 3)
#         v = pcl_source[:, :num_points, None, :3] - pcl_nn       # (1, N, K, 3)
#         v1 = v[:,:,:8,:].mean(-2)                               # (1, N, 3)
#         v2 = v[:,:,:4,:].mean(-2)                               # (1, N, 3)
#         v3 = v[:,:,0,:]                                         # (1, N, 3)

#         pcl_target = torch.cat((pcl_nn[:,:,0,:], pcl_source[:, num_points:, :]), dim=-2)

#         loss_reg1 = 10 * (self.sd[:, num_points:, :]**2).mean()
#         loss_reg2 = 10 * (self.sd1**2).mean() #+ 10 * (self.sd2**2).mean()

#         weight = torch.exp(-60 * torch.abs(self.sd)).squeeze()      # (N,)

#         loss_v1 = torch.linalg.norm((v1 - (self.sd * self.grad_norm)[:, :num_points, :]), ord=2, dim=-1).mean()
#         loss_v2 = torch.linalg.norm((v2 - (self.sd * self.grad_norm)[:, :num_points, :]), ord=2, dim=-1).mean()
#         loss_v3 = torch.linalg.norm((v3 - (self.sd * self.grad_norm)[:, :num_points, :]), ord=2, dim=-1).mean()

#         pcl_source_new = pcl_source - self.sd * self.grad_norm - self.sd1 * self.grad_norm1 #- self.sd2 * self.grad_norm2
#         loss_d = 0.3 * torch.linalg.norm((pcl_source_new - pcl_target), ord=2, dim=-1).mean()

#         cos_ang = cos_angle(self.grad_norm[0, :, :], self.grad_norm1[0, :, :])  # (N,)
#         # cos_ang1 = cos_angle(self.grad_norm[0, :, :], self.grad_norm2[0, :, :])
#         loss_con = 0.01 * (weight * (1 - cos_ang)).mean() #+ 0.01 * (weight * (1 - cos_ang1)).mean()

#         # loss_sd = 0.01 * torch.clamp(torch.abs(self.sd + self.sd1)[:, :num_points, :] - torch.linalg.norm(v3, ord=2, dim=-1), min=0.0).mean()

#         loss_tuple = (loss_v1, loss_v2, loss_v3, loss_d, loss_reg1, loss_reg2, loss_con, loss_sd)
#         loss_sum = sum(loss_tuple)
#         return loss_sum, loss_tuple


# # --- Example Usage ---
# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Example point cloud (replace with your actual data)
#     P_full = torch.randn(2048, 3, device=device) * 2.0 # Example full shape

#     # Example query points
#     x_query = torch.randn(1024, 3, device=device) # Batch of points to query

#     # Instantiate the network
#     network = AttentionNeuralGF(
#         d_feature=128,
#         d_mlp_hidden=128,
#         n_mlp_layers=3,
#         n_attn_heads=4,
#         n_global_attn_layers=1,
#         k_neighbors=32,
#         n_subsample_global=256
#     ).to(device)

#     # --- IMPORTANT: Optimization Context ---
#     # This network is designed to be optimized PER SHAPE.
#     # You would typically do something like this (pseudo-code):
#     #
#     # optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
#     # for step in range(num_optimization_steps):
#     #     optimizer.zero_grad()
#     #     # Sample query points Q and maybe surface points G from P_full
#     #     sampled_queries = ... # sample points around P_full
#     #     sampled_surface = ... # sample points from P_full
#     #
#     #     # Predict SDF for sampled points
#     #     predicted_sdf = network(sampled_queries, P_full)
#     #     # Predict normals if needed for loss
#     #     predicted_normals = network.get_gradient(sampled_surface, P_full)
#     #
#     #     # Calculate NeuralGF losses (using predicted_sdf, predicted_normals,
#     #     # ground truth points P_full, maybe sampled_surface normals if estimated differently)
#     #     loss = compute_neuralgf_loss(...) # This requires implementing NeuralGF's loss
#     #
#     #     loss.backward()
#     #     optimizer.step()
#     #
#     # print(f"Finished optimizing network for the shape P_full")
#     #
#     # --- After optimization, use the network ---

#     # Perform a forward pass to get signed distance
#     network.eval() # Set to evaluation mode if using dropout etc.
#     with torch.no_grad(): # No need for gradients during inference
#         predicted_sdf = network(x_query, P_full)

#     print(f"Predicted SDF shape: {predicted_sdf.shape}") # Should be (1024, 1)

#     # Get the normals (gradients) at the query points
#     # No need for torch.no_grad() here as get_gradient handles requires_grad
#     predicted_normals = network.get_gradient(x_query, P_full)

#     print(f"Predicted Normals shape: {predicted_normals.shape}") # Should be (1024, 3)

#     # Example: Check if normals are unit length
#     print(f"Normals norms (mean): {torch.norm(predicted_normals, dim=-1).mean().item()}")



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from pytorch3d.ops import knn_gather, knn_points
# import numpy as np

# # Helper function (same as before)
# def cos_angle(v1, v2):
#     return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)

# # --- NEW: Local Feature Extractor ---
# class LocalFeatureExtractor(nn.Module):
#     """
#     Extracts features from local K-nearest neighbors.
#     Input: Relative neighbor coordinates (B, N, K, 3)
#     Output: Local feature vector per point (B, N, D_local_feat)
#     """
#     def __init__(self, k, d_local_feat=128):
#         super().__init__()
#         self.k = k
#         self.d_local_feat = d_local_feat

#         # Simple PointNet-like MLP applied to each relative neighbor vector
#         # Use Conv2d with kernel_size (1, 1) for shared MLP effect
#         self.mlp1 = nn.Conv2d(3, 64, 1)
#         self.mlp2 = nn.Conv2d(64, self.d_local_feat, 1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(self.d_local_feat)

#     def forward(self, relative_neighbors):
#         """
#         relative_neighbors: (B, N, K, 3) tensor of (p_neighbor - p_center)
#         """
#         # Permute for Conv2d: (B, 3, K, N)
#         x = relative_neighbors.permute(0, 3, 2, 1)
        
#         x = F.relu(self.bn1(self.mlp1(x))) # (B, 64, K, N)
#         x = F.relu(self.bn2(self.mlp2(x))) # (B, D_local_feat, K, N)

#         # Max pooling over neighbors (K dimension)
#         # Output shape: (B, D_local_feat, N)
#         x = torch.max(x, dim=2, keepdim=False)[0] 

#         # Permute back: (B, N, D_local_feat)
#         x = x.permute(0, 2, 1)
#         return x

# class MLPNet_linear(nn.Module):
#     def __init__(self,
#                  d_in=3,
#                  d_mid=256,
#                  d_out=1,
#                  n_mid=8,
#                  d_local_feat=128,
#                  bias=0.5,
#                  geometric_init=True,
#                  weight_norm=False,
#                  inside_grad=True,
#                  film_layers=[1, 3, 5, 7]
#             ):
#         super(MLPNet_linear, self).__init__()
#         assert n_mid > 3
#         self.d_local_feat = d_local_feat
#         self.film_layers = set(film_layers)

#         dims = [d_in] + [d_mid for _ in range(n_mid)] + [d_out]
#         self.num_layers = len(dims)
#         # Skip connection is applied *before* layer index skip_in[0]
#         self.skip_in = [n_mid // 2]

#         self.layer_out_dims = {} # Store actual output dimensions

#         # --- Main MLP Layers ---
#         for l in range(0, self.num_layers - 1):
#             # --- Corrected: Use simple dims[l] for input dimension ---
#             # The concatenation for skip connection happens dynamically in forward
#             in_dim = dims[l]

#             # Determine the output dimension for lin[l]
#             # Adjust output if the *next* layer is the target for skip input
#             if l + 1 in self.skip_in:
#                 out_dim = dims[l + 1] - d_in
#             else:
#                 out_dim = dims[l + 1]

#             # Store the actual calculated output dimension for lin[l]
#             self.layer_out_dims[l] = out_dim

#             # Use the simple in_dim for layer creation
#             lin = nn.Linear(in_dim, out_dim)

#             # --- Geometric Initialization (using correct in_dim) ---
#             if geometric_init:
#                 if l == self.num_layers - 2: # Final layer before output
#                     if inside_grad:
#                         nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
#                         nn.init.constant_(lin.bias, bias)
#                     else:
#                         nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
#                         nn.init.constant_(lin.bias, -bias)
#                 # --- Simpler Init for other layers ---
#                 # (Original code had more complex init logic based on skips,
#                 # restoring a simpler one here for clarity, review if needed)
#                 elif l < self.num_layers - 2 :
#                     nn.init.normal_(lin.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(out_dim))
#                     nn.init.constant_(lin.bias, 0.0)
#             # --- End Geometric Initialization ---

#             if weight_norm:
#                 lin = nn.utils.weight_norm(lin)
#             setattr(self, "lin" + str(l), lin)

#         # --- FiLM Modulation Networks ---
#         for l in self.film_layers:
#             if l < self.num_layers - 1: # Don't modulate output layer
#                 # Use the ACTUAL output dimension of lin[l] stored earlier
#                 actual_layer_out_dim = self.layer_out_dims[l]
#                 mod_net = nn.Linear(self.d_local_feat, actual_layer_out_dim * 2)

#                 # Initialize FiLM layers
#                 nn.init.constant_(mod_net.bias[:actual_layer_out_dim], 1.0) # Gamma bias
#                 nn.init.constant_(mod_net.bias[actual_layer_out_dim:], 0.0) # Beta bias
#                 nn.init.zeros_(mod_net.weight)
#                 setattr(self, "mod" + str(l), mod_net)

#     # --- forward and gradient methods remain the same as the previous version ---
#     def forward(self, pos, local_feat):
#         """
#         pos: (*, N, 3) coordinates
#         local_feat: (*, N, D_local_feat) local features for FiLM
#         """
#         x = pos # Input coordinates
#         x_input = pos # Store original input for skip connection

#         for l in range(0, self.num_layers - 1):
#             lin = getattr(self, "lin" + str(l))

#             # --- Handle Skip Connection ---
#             # Concatenate x_input just BEFORE feeding into layer lin[l] if l is a skip_in target layer index
#             if l in self.skip_in:
#                 x = torch.cat([x, x_input], dim=-1)
#                 # x = x / np.sqrt(2) # Optional normalization

#             # --- Apply Linear Layer ---
#             # Now x has the correct dimension expected by lin (after potential concat)
#             # because lin was created with in_features = dims[l]
#             x = lin(x)

#             # --- Apply Activation (except for output layer) ---
#             if l < self.num_layers - 2:
#                 x = F.relu(x)

#                 # --- Apply FiLM post-activation ---
#                 if l in self.film_layers:
#                     mod_net = getattr(self, "mod" + str(l))
#                     gamma_beta = mod_net(local_feat)
#                     gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

#                     # Apply FiLM - shapes should match now
#                     x = gamma * x + beta

#         return x # Output SDF value (*, N, 1)

#     def gradient(self, pos, local_feat):
#         """
#         pos: (*, N, 3), with requires_grad=True
#         local_feat: (*, N, D_local_feat), conditioning features (no grad needed w.r.t pos)
#         """
#         local_feat = local_feat.detach()
#         y = self.forward(pos, local_feat)
#         grad_outputs = torch.ones_like(y, requires_grad=False, device=y.device)
#         grad_pos = torch.autograd.grad(
#             outputs=y,
#             inputs=pos,
#             grad_outputs=grad_outputs,
#             create_graph=True,
#             retain_graph=True,
#             only_inputs=True
#         )[0]
#         grad_norm = F.normalize(grad_pos, dim=-1)
#         return y, grad_norm
    
#     # --- MODIFIED: Main Network Class ---
# class Network(nn.Module):
#     def __init__(self, num_points, num_knn, d_local_feat=128): # Added d_local_feat
#         super(Network, self).__init__()
#         self.num_points = num_points
#         self.num_knn = num_knn
#         self.d_local_feat = d_local_feat
#         self.num_iter = 2 # Number of refinement iterations

#         # Instantiate the local feature extractor
#         self.local_feature_extractor = LocalFeatureExtractor(k=self.num_knn, d_local_feat=self.d_local_feat)

#         # Instantiate the FiLM-conditioned MLP
#         self.net = MLPNet_linear(d_in=3, d_mid=256, d_out=1, n_mid=8, d_local_feat=self.d_local_feat)

#     def _get_local_features(self, pcl_ref, pcl_query, k):
#         """ Helper to compute KNN and extract local features """
#         # Find K nearest neighbors in pcl_ref for each point in pcl_query
#         knn_out = knn_points(pcl_query, pcl_ref, K=k, return_nn=True)
#         knn_idx = knn_out.idx # (B, N_query, K)
#         knn_neighbors = knn_out.knn # (B, N_query, K, 3)

#         # Compute relative coordinates
#         relative_neighbors = knn_neighbors - pcl_query.unsqueeze(2)

#         # Extract local features
#         local_feat = self.local_feature_extractor(relative_neighbors)
#         return local_feat, knn_idx # Return knn_idx if needed elsewhere

#     def forward(self, pcl_source):
#         """
#         pcl_source: (*, N, 3) input point cloud coordinates
#         """
#         self.sd_all = []
#         self.grad_all = []
        
#         # The `pcl_source` tensor will be modified in the loop,
#         # so ensure requires_grad is set appropriately.
#         pcl_current = pcl_source.clone().detach().requires_grad_(True)

#         # Initial dummy values for the loop start
#         # Use shapes consistent with batches if pcl_source has a batch dim
#         batch_dim = pcl_source.shape[:-2]
#         num_pts = pcl_source.shape[-2]
#         sd_temp = torch.zeros(*batch_dim, num_pts, 1, device=pcl_source.device, dtype=pcl_source.dtype)
#         grad_temp = torch.zeros_like(pcl_source)


#         with torch.enable_grad(): # Ensure grad is enabled for the loop
#             for i in range(self.num_iter):
#                 # --- Compute local features for the CURRENT point positions ---
#                 # We use pcl_source (original points) as the reference set
#                 # and pcl_current (potentially shifted points) as the query set.
#                 # Detach pcl_source to avoid gradients flowing back through the reference points.
#                 local_feat, _ = self._get_local_features(
#                     pcl_ref=pcl_source.detach(), 
#                     pcl_query=pcl_current, 
#                     k=self.num_knn
#                 )

#                 # Update position based on previous iteration's estimate
#                 # Make sure not to update in-place if pcl_current is needed later
#                 pcl_proj_prev_iter = pcl_current - sd_temp * grad_temp
                
#                 # Re-enable grad for the projected points for the gradient calculation
#                 pcl_proj_prev_iter.requires_grad_(True)
                
#                 # --- Get SDF and Gradient from FiLM network ---
#                 sd_temp, grad_temp = self.net.gradient(pcl_proj_prev_iter, local_feat)

#                 # Store results for loss computation and final aggregation
#                 self.sd_all.append(sd_temp)
#                 self.grad_all.append(grad_temp)

#                 # Prepare for next iteration: update current point cloud estimate
#                 # Detach results before using them for the next position update
#                 # to avoid overly complex graph backpropagation across iterations.
#                 sd_temp = sd_temp.detach()
#                 grad_temp = grad_temp.detach()
#                 pcl_current = pcl_proj_prev_iter.detach() # Use the point location *before* grad was computed

#                 # Store specific iteration results if needed (optional)
#                 if i == 0:
#                     self.sd = sd_temp
#                     self.grad_norm = grad_temp
#                 elif i == 1:
#                     self.sd1 = sd_temp
#                     self.grad_norm1 = grad_temp
#                 # Add more elif blocks if self.num_iter > 2

#             # Final aggregated normal (same logic as before)
#             self.grad_sum = F.normalize(sum(self.grad_all), dim=-1)

#         # Detach the final output if it's not meant to be part of a larger graph
#         return self.grad_sum.detach()


#     def get_loss(self, pcl_raw=None, pcl_source=None, knn_idx=None):
#         """
#         Calculates the unsupervised loss.
#         Requires knn_idx for pcl_raw neighbors of pcl_source[:num_points].
#         NOTE: This function remains structurally the same, but the values
#               (sd, grad_norm, sd1, grad_norm1) are now computed by the
#               FiLM-conditioned network.

#         pcl_raw: (B, M, 3), M >= N, reference point cloud (potentially denser)
#         pcl_source: (B, N+n, 3), input points used in forward pass
#         knn_idx: (B, N, K) indices of nearest neighbors in pcl_raw for the
#                  first N points in pcl_source.
#         """
#         if knn_idx is None:
#              # If not provided, compute KNN for the loss terms
#              # Use only the first num_points from pcl_source for geometric consistency loss
#              _, knn_idx = knn_points(pcl_source[:, :self.num_points, :], pcl_raw, K=self.num_knn)


#         num_points = self.num_points
#         _device, _dtype = pcl_source.device, pcl_source.dtype
#         loss_d = torch.zeros(1, device=_device, dtype=_dtype)
#         loss_v1 = torch.zeros(1, device=_device, dtype=_dtype)
#         loss_v2 = torch.zeros(1, device=_device, dtype=_dtype)
#         loss_v3 = torch.zeros(1, device=_device, dtype=_dtype)
#         loss_reg1 = torch.zeros(1, device=_device, dtype=_dtype)
#         loss_reg2 = torch.zeros(1, device=_device, dtype=_dtype)
#         loss_con = torch.zeros(1, device=_device, dtype=_dtype)
#         loss_sd = torch.zeros(1, device=_device, dtype=_dtype) # Example placeholder, unused in original

#         # Gather neighbors using the provided or computed knn_idx
#         pcl_nn = knn_gather(pcl_raw, knn_idx)                   # (B, N, K, 3)

#         # Compute displacement vectors (relative positions)
#         # Only for the first num_points which correspond to the original surface points
#         v = pcl_source[:, :num_points, None, :3] - pcl_nn       # (B, N, K, 3)
#         v1 = v[:,:,:8,:].mean(-2)                               # (B, N, 3) - Mean of nearest 8 neighbors
#         v2 = v[:,:,:4,:].mean(-2)                               # (B, N, 3) - Mean of nearest 4 neighbors
#         v3 = v[:,:,0,:]                                         # (B, N, 3) - Vector to the single nearest neighbor

#         # Target points for the projection loss (nearest neighbor + augmented points)
#         # Assuming pcl_source[:, num_points:, :] are the augmented 'noise' points
#         pcl_target = torch.cat((pcl_nn[:,:,0,:], pcl_source[:, num_points:, :]), dim=-2) # Check dim carefully, should be dim=1 if B=1

#         # --- Regularization Losses ---
#         # Penalize SDF magnitude for augmented points (should be near zero)
#         loss_reg1 = 10 * (self.sd[:, num_points:, :]**2).mean()
#         # Penalize SDF magnitude for all points after the first iteration (should be closer to zero)
#         loss_reg2 = 10 * (self.sd1**2).mean() # Add sd2 if num_iter > 1

#         # --- Geometric Consistency Losses ---
#         # Compare estimated projection (sd * normal) to mean displacements
#         # Only apply to the first N points
#         loss_v1 = torch.linalg.norm((v1 - (self.sd[:, :num_points, :] * self.grad_norm[:, :num_points, :])), ord=2, dim=-1).mean()
#         loss_v2 = torch.linalg.norm((v2 - (self.sd[:, :num_points, :] * self.grad_norm[:, :num_points, :])), ord=2, dim=-1).mean()
#         loss_v3 = torch.linalg.norm((v3 - (self.sd[:, :num_points, :] * self.grad_norm[:, :num_points, :])), ord=2, dim=-1).mean()

#         # --- Projection Loss ---
#         # Calculate the final projected point after all iterations
#         # Ensure all sd and grad_norm tensors used here have the correct shape (B, N+n, ...)
#         pcl_source_new = pcl_source.clone() # Start from original points for projection
#         for i in range(self.num_iter):
#              # Make sure sd_all and grad_all have consistent shapes
#              sd_i = self.sd_all[i]
#              grad_i = self.grad_all[i]
#              if sd_i.shape[1] != pcl_source_new.shape[1]: # Handle potential shape mismatch if only N points were processed somewhere
#                  # This indicates an issue needs fixing upstream in forward/gradient handling
#                  # For now, maybe pad or slice, but ideally shapes should match
#                  print(f"Warning: Shape mismatch in loss calculation iter {i}. sd: {sd_i.shape}, grad: {grad_i.shape}, pcl: {pcl_source_new.shape}")
#                  # Example fix: Assuming sd/grad correspond to all N+n points
#                  pass # Assume shapes match for now

#              pcl_source_new = pcl_source_new - sd_i * grad_i


#         # Compare final projected points to target points
#         loss_d = 0.3 * torch.linalg.norm((pcl_source_new - pcl_target), ord=2, dim=-1).mean()

#         # --- Normal Consistency Loss ---
#         # Encourage normals from different iterations to be consistent
#         weight = torch.exp(-60 * torch.abs(self.sd[:,:num_points,:])).squeeze(-1)      # (B, N) - Use only first N points
#         # Ensure grad_norm and grad_norm1 correspond to the first N points if needed
#         cos_ang = cos_angle(self.grad_norm[:, :num_points, :].reshape(-1, 3),
#                             self.grad_norm1[:, :num_points, :].reshape(-1, 3)) # (B*N,)
#         cos_ang = cos_ang.reshape(pcl_source.shape[0], num_points) # Reshape back to (B, N)

#         loss_con = 0.01 * (weight * (1 - cos_ang)).mean()
#         # Add consistency with grad_norm2 if num_iter > 2

#         # loss_sd = ... # Optional additional loss terms

#         loss_tuple = (loss_v1, loss_v2, loss_v3, loss_d, loss_reg1, loss_reg2, loss_con, loss_sd)
#         loss_sum = sum(loss.mean() for loss in loss_tuple) # Ensure reduction if any loss is multi-element
#         return loss_sum, loss_tuple