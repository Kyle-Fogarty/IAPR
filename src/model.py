import torch
import torch.nn as nn
import math
from pytorch3d.ops import sample_farthest_points, knn_points
from src.nn import MLPNet_linear

import sys
sys.path.append('.')
from src.nn import *
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



class ImplicitAttentionFields(nn.Module):
    """
    Hybrid Neural Field with dynamic patch encoding and cross-attention decoding.
    
    This model:
    1. Dynamically selects anchor points from input point clouds
    2. Forms local patches around anchor points
    3. Encodes patches into latent representations
    4. Uses transformers to process these representations
    5. Decodes query points using cross-attention with the latent set
    """
    def __init__(
        self,
        num_anchor_points=16,    # Number of anchor points (M)
        num_neighbors=512,         # Number of neighbors per patch (K)
        latent_dim=128,           # Latent dimension (C)
        num_encoder_blocks=4,     # Number of transformer blocks
        pos_encoding_freqs=6,     # Number of positional encoding frequencies
        mlp_hidden_dim=128,       # Hidden dimension in MLPs
        num_mlp_layers=4,         # Number of MLP layers in decoder
        num_attn_heads=8,         # Number of attention heads
        dropout_rate=0.0          # Dropout rate
    ):
        """
        Initialize the Hybrid Neural Field model.
        """
        super().__init__()
        self.num_iter = 2
        self.num_points = 5000
        self.num_knn = 64
        # Configuration
        self.num_anchor_points = num_anchor_points
        self.num_neighbors = num_neighbors
        self.latent_dim = latent_dim
        
        # --- Positional Encoding ---
        self.positional_encoder = PositionalEncoder(
            input_dim=3, 
            num_freqs=pos_encoding_freqs
        )
        pe_output_dim = self.positional_encoder.output_dim
        
        # --- Encoder Components ---
        # Patch processor
        self.mini_pointnet = MiniPointNet(
            input_dim=3, 
            output_dim=latent_dim
        )
        
        # Positional encoding projector
        self.anchor_pe_proj = nn.Linear(pe_output_dim, latent_dim)
        
        # Transformer blocks for encoding
        encoder_ff_dim = mlp_hidden_dim * 2
        self.encoder_transformer_blocks = nn.ModuleList([
            TransformerBlock(latent_dim, num_attn_heads, encoder_ff_dim, dropout_rate)
            for _ in range(num_encoder_blocks)
        ])
        
        # --- Decoder Components ---
        # Query point projector
        self.decoder_query_proj = nn.Linear(pe_output_dim, latent_dim)
        self.decoder_query_proj_linear = nn.Linear(3, latent_dim)
        # Cross-attention mechanism
        self.decoder_cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, 
            num_heads=num_attn_heads, 
            dropout=dropout_rate, 
            batch_first=True
        )
        
        # Post-attention processing
        self.decoder_norm1 = nn.LayerNorm(latent_dim)
        self.decoder_dropout = nn.Dropout(dropout_rate)
        self.decoder_ffn = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden_dim * 2), 
            nn.GELU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim * 2, latent_dim)
        )
        self.decoder_norm2 = nn.LayerNorm(latent_dim)
        
        # Output MLP
        mlp_layers = []
        current_dim = latent_dim
        for _ in range(num_mlp_layers - 1):
            mlp_layers.extend([
                nn.Linear(current_dim, mlp_hidden_dim), 
                nn.GELU()
            ])
            current_dim = mlp_hidden_dim
        mlp_layers.append(nn.Linear(current_dim, 1))
        self.mlp_decoder = nn.Sequential(*mlp_layers)
        self.mlp_decoder = MLPNet_linear(d_in = latent_dim)

        self.current_iter = 0
        self.weight = 0.0
        self.anchor_points = None
        self.patches = None
        self.point_cloud = None
        self.procesor = None
        
        # Initialize learnable latent set separately
        self.init_latent_set()
        self.check_orthogonality()

    # def init_latent_set(self):
    #     """Initialize the learnable latent set separately"""
    #     self.learnable_latent_set = nn.Parameter(torch.randn(1, self.num_anchor_points, self.latent_dim))
            # self.init_latent_set()

    def init_latent_set(self):
        """Initialize the learnable latent set with orthogonal vectors."""
        # Create an empty tensor for the parameter first
        # The shape is (1, num_anchor_points, latent_dim)
        # We want the `num_anchor_points` vectors (each of `latent_dim`) to be orthogonal.
        # nn.init.orthogonal_ works on 2D tensors.
        # So we'll initialize a 2D tensor of shape (num_anchor_points, latent_dim)
        # and then unsqueeze it.

        if self.num_anchor_points > self.latent_dim:
            print(
                f"Warning: num_anchor_points ({self.num_anchor_points}) > latent_dim ({self.latent_dim}). "
                "The initialized vectors will span an orthogonal basis, "
                "but will not be mutually orthogonal themselves."
            )
        
        # Create a 2D tensor to be made orthogonal
        # This tensor will represent our `num_anchor_points` vectors, each of `latent_dim`
        orthogonal_matrix = torch.empty(self.num_anchor_points, self.latent_dim)
        
        # Apply orthogonal initialization
        # gain=1.0 is common for linear layers without specific activations,
        # or if activations are handled by subsequent layers.
        nn.init.orthogonal_(orthogonal_matrix, gain=1.0) 
        
        # Add the batch dimension (size 1) and wrap in nn.Parameter
        self.learnable_latent_set = nn.Parameter(orthogonal_matrix.unsqueeze(0))

    def check_orthogonality(self):
        """Helper to verify orthogonality (for num_anchor_points <= latent_dim)."""
        if self.num_anchor_points > self.latent_dim:
            print("Orthogonality check skipped: num_anchor_points > latent_dim.")
            # Check if columns are orthogonal instead
            # Squeeze the batch dimension
            mat = self.learnable_latent_set.squeeze(0) # Shape: [num_anchor_points, latent_dim]
            # Check column orthogonality (latent_dim vectors of size num_anchor_points)
            dot_products_cols = mat.T @ mat
            identity_cols = torch.eye(self.latent_dim)
            is_orthogonal_cols = torch.allclose(dot_products_cols, identity_cols, atol=1e-6) # Looser tolerance due to numerical precision
            print(f"  Columns are orthogonal: {is_orthogonal_cols}")
            # You can also check singular values:
            s = torch.linalg.svdvals(mat)
            print(f"  Singular values (should be close to 1.0 for the {self.latent_dim} non-zero ones): {s}")

            return is_orthogonal_cols # Or some other relevant metric
        
        # Squeeze the batch dimension
        mat = self.learnable_latent_set.squeeze(0) # Shape: [num_anchor_points, latent_dim]
        
        # Compute dot products: mat @ mat.T
        # Resulting shape: [num_anchor_points, num_anchor_points]
        # This should be close to an identity matrix if rows are orthogonal
        dot_products = mat @ mat.T
        
        # Create an identity matrix for comparison
        identity = torch.eye(self.num_anchor_points)
        
        # Check if dot_products is close to identity
        # Using a small tolerance (atol) for floating point comparisons
        is_orthogonal = torch.allclose(dot_products, identity, atol=1e-6) # Orthogonal init makes them unit norm too
        
        print(f"Shape of latent set: {self.learnable_latent_set.shape}")
        print(f"Shape of matrix for dot product: {mat.shape}")
        print(f"Dot product matrix (should be ~Identity):\n{dot_products}")
        print(f"Is the latent set orthogonal (rows)? {is_orthogonal}")
        
        # You can also check singular values of the matrix 'mat'.
        # For an orthogonal matrix (where num_rows <= num_cols),
        # the singular values corresponding to the rows should be 1.0.
        s = torch.linalg.svdvals(mat)
        print(f"Singular values (first {self.num_anchor_points} should be ~1.0): {s}")

        return is_orthogonal
    def _fixed_plc_encode(self):
        
        

        if self.anchor_points is None:
            batch_size, num_points_in, _ = self.point_cloud.shape
            # Adjust M if point cloud is too small
            M = min(self.num_anchor_points, num_points_in)
            K = self.num_neighbors
            self.M = M
            self.K = K
            # 1. Select anchor points via farthest point sampling
            anchor_points, _ = farthest_point_sample(self.point_cloud, M)  # [B, M, 3]
            
            # 2. Form patches around anchor points using KNN
            patches, _ = knn(self.point_cloud, anchor_points, K)  # [B, M, K, 3]

            self.anchor_points = anchor_points
            self.patches = patches

        batch_size, num_points_in, _ = self.point_cloud.shape
        # 3. Encode patches with MiniPointNet
        # Reshape for processing: [B, M, K, 3] -> [B*M, K, 3] -> [B*M, 3, K]
        patches_flat = self.patches.view(batch_size * self.M, self.K, 3)
        patches_flat_t = patches_flat.transpose(1, 2)
        
        # Get patch features: [B*M, 3, K] -> [B*M, C] -> [B, M, C]
        patch_features_flat = self.mini_pointnet(patches_flat_t)
        patch_features = patch_features_flat.view(batch_size, self.M, self.latent_dim)
        
        # 4. Encode anchor point positions
        anchor_pe      = self.positional_encoder(self.anchor_points)  # [B, M, PE_Dim]
        anchor_pe_proj = self.anchor_pe_proj(anchor_pe)     # [B, M, C]
        
        # 5. Combine patch features with positional encoding
        fused_features = patch_features + anchor_pe_proj    # [B, M, C]
        
        # 6. Process with transformer blocks
        latent_set = fused_features
        for block in self.encoder_transformer_blocks:
            latent_set = block(latent_set)  # [B, M, C]
            
        return latent_set

    def _encode(self, point_cloud):
        """
        Encode point cloud into a latent representation.
        
        Args:
            point_cloud: Input point cloud, shape [B, N, 3]
            
        Returns:
            latent_set: Encoded latent representations, shape [B, M, C]
        """
        batch_size, num_points_in, _ = point_cloud.shape
        # Adjust M if point cloud is too small
        M = min(self.num_anchor_points, num_points_in)
        K = self.num_neighbors
        
        # 1. Select anchor points via farthest point sampling
        anchor_points, _ = farthest_point_sample(point_cloud, M)  # [B, M, 3]
        
        # 2. Form patches around anchor points using KNN
        patches, _ = knn(point_cloud, anchor_points, K)  # [B, M, K, 3]
        
        # 3. Encode patches with MiniPointNet
        # Reshape for processing: [B, M, K, 3] -> [B*M, K, 3] -> [B*M, 3, K]
        patches_flat = patches.view(batch_size * M, K, 3)
        patches_flat_t = patches_flat.transpose(1, 2)
        
        # Get patch features: [B*M, 3, K] -> [B*M, C] -> [B, M, C]
        patch_features_flat = self.mini_pointnet(patches_flat_t)
        patch_features = patch_features_flat.view(batch_size, M, self.latent_dim)
        
        # 4. Encode anchor point positions
        anchor_pe = self.positional_encoder(anchor_points)  # [B, M, PE_Dim]
        anchor_pe_proj = self.anchor_pe_proj(anchor_pe)     # [B, M, C]
        
        # 5. Combine patch features with positional encoding
        fused_features = patch_features + anchor_pe_proj    # [B, M, C]
        
        # 6. Process with transformer blocks
        latent_set = fused_features
        for block in self.encoder_transformer_blocks:
            latent_set = block(latent_set)  # [B, M, C]
            
        return latent_set

    def _decode(self, query_points, latent_set):
        """
        Decode field values at query points using the latent representation.
        
        Args:
            query_points: Query points, shape [B, Q, 3]
            latent_set: Encoded latent representations, shape [B, M, C]
            
        Returns:
            field_values: Decoded field values, shape [B, Q, 1]
        """
        batch_size, num_query_pts, _ = query_points.shape
        
        # 1. Encode query points with positional encoding
        query_embedded = self.positional_encoder(query_points)     # [B, Q, PE_Dim]
        decoder_queries = self.decoder_query_proj(query_embedded)  # [B, Q, C]
        
        decoder_queries_linear = self.decoder_query_proj_linear(query_points)  # [B, Q, C]
        if latent_set is not None:
            # 2. Cross-attention between query points and latent set
            attn_output, cross_attn_weights = self.decoder_cross_attn(query  =   decoder_queries, 
                                                    key    =   latent_set, 
                                                    value =   latent_set)  # [B, Q, C]
            self.cross_attn_weights_inference = cross_attn_weights
        else:
            attn_output = decoder_queries
            
        # 3. Post-attention processing
        # Residual connection and normalization
        # Calculate weight based on iteration
        # if self.current_iter < 10000:
        #     weight = 0.0
        #     self.weight = weight
        # else:
        #     # Linear increase from 0 to 1 between iterations 10000-20000
        #     weight = min(1.0, (self.current_iter - 10000) / 10000)
        #     self.weight = weight
        weight = self.weight
        self.current_iter += 1
        attn_output = (1-weight)*decoder_queries_linear + weight*self.decoder_dropout(attn_output)
        # attn_output_norm = self.decoder_norm1(attn_output)
        
        # Feedforward network
        # ffn_output = self.decoder_ffn(attn_output)
        # processed_features = attn_output + self.decoder_dropout(ffn_output)
        # processed_features = self.decoder_norm2(processed_features)  # [B, Q, C]
        
        # 4. Final MLP prediction
        field_values = self.mlp_decoder(attn_output)  # [B, Q, 1]
        
        return field_values

    def forward(self, point_cloud, query_points):
        """
        End-to-end forward pass.
        
        Args:
            point_cloud: Input point cloud, shape [B, N, 3]
            query_points: Query points, shape [B, Q, 3]
            
        Returns:
            field_values: Predicted field values, shape [B, Q, 1]
        """
        if self.weight == 0.0:
            latent_set = None
        else:
            if self.point_cloud is None:
                self.point_cloud = point_cloud
            # # 1. Encode the input point cloud
            # latent_set = self._encode(point_cloud)  # [B, M, C]
            # latent_set = None
            # latent_set = self._fixed_plc_encode()
        latent_set = self.learnable_latent_set
        # 2. Decode field values at query points
        field_values = self._decode(query_points, latent_set)  # [B, Q, 1]
        
        return field_values
    
    def gradient(self, point_cloud, query_points):

        query_points = query_points.requires_grad_(True)
        scalar_val = self.forward(point_cloud, query_points)

        scalar_val.requires_grad_(True)

        grad_outputs = torch.ones_like(scalar_val, requires_grad=False, device=scalar_val.device)
        grad_out = torch.autograd.grad(outputs=scalar_val,
                                    inputs=query_points,
                                    grad_outputs=grad_outputs,
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]
        grad_norm = F.normalize(grad_out, dim=-1)     # (*, N, 3)
        # gradient_field = None
        
        # query_points1 = point_cloud.clone().requires_grad_(True)
        # scalar_val = self.forward(point_cloud, query_points1)

        # scalar_val.requires_grad_(True)

        # grad_outputs = torch.ones_like(scalar_val, requires_grad=False, device=scalar_val.device)
        # grad_out = torch.autograd.grad(outputs=scalar_val,
        #                             inputs=query_points1,
        #                             grad_outputs=grad_outputs,
        #                             create_graph=True,
        #                             retain_graph=True,
        #                             only_inputs=True)[0]
        # grad_norm = F.normalize(grad_out, dim=-1)     # (*, N, 3)
        # gradient_field = None
        

        # scalar_val, grad_norm = self.processor(query_points.squeeze(), self.point_cloud.squeeze(), grad_norm.squeeze(), compute_gradient=True)
        # scalar_val, grad_norm = scalar_val.unsqueeze(0).unsqueeze(2), grad_norm.unsqueeze(0)
        return scalar_val, grad_norm
    def run_forward(self, plc_raw, pcl_source):
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
                # print(pcl_source.shape)
                # print(sd_temp.shape)
                # print(grad_temp.shape)
                pcl_source = pcl_source - sd_temp * grad_temp

                
                sd_temp, grad_temp = self.gradient(plc_raw, pcl_source)     # (*, N, 1), (*, N, 3)
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


        # torch.nn.functional.co self.learnable_latent_set
        return loss_sum, loss_tuple

