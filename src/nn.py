"""
Hybrid Neural Field Implementation for 3D Point Cloud Processing
----------------------------------------------------------------
This module implements a hybrid neural field approach that combines patch-based 
point cloud encoding with transformer-based processing and cross-attention decoding.
"""

import torch
import torch.nn as nn
import math
from pytorch3d.ops import sample_farthest_points, knn_points
# from src.nn import MLPNet_linear
# ---------------------- Point Cloud Operations ----------------------

def farthest_point_sample(xyz, npoint):
    """
    Sample points using the farthest point sampling algorithm.
    
    Args:
        xyz: Point cloud data, shape [B, N, 3]
        npoint: Number of points to sample
        
    Returns:
        sampled_points: Sampled points, shape [B, npoint, 3]
        index: Indices of sampled points
    """
    sampled_points, index = sample_farthest_points(
        points=xyz, 
        K=npoint, 
        random_start_point=True
    )
    return sampled_points, index


def knn(ref_points, query_points, k):
    """
    Find k-nearest neighbors for query points in reference points.
    
    Args:
        ref_points: Reference points, shape [B, N, 3]
        query_points: Query points, shape [B, M, 3]
        k: Number of neighbors to find
        
    Returns:
        neighbors: K nearest neighbor points, shape [B, M, k, 3]
        indices: Indices of K nearest neighbors, shape [B, M, k]
    """
    # Get KNN indices and distances
    distances, indices, _ = knn_points(
        p1=query_points,
        p2=ref_points,
        K=k,
        return_nn=True
    )
    
    # Get the actual neighbor points using the indices
    B, M, _ = query_points.shape
    C = ref_points.shape[-1]
    
    # Expand indices for gathering
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, C)
    # Expand reference points for gathering
    ref_expanded = ref_points.unsqueeze(1).expand(-1, M, -1, -1)
    # Gather neighbors
    neighbors = torch.gather(ref_expanded, 2, indices_expanded)
    
    return neighbors, indices


# ---------------------- Basic Network Modules ----------------------

class PositionalEncoder(nn.Module):
    """
    Sinusoidal Positional Encoding for 3D coordinates.
    
    Transforms 3D coordinates into a higher-dimensional representation
    using sinusoidal functions at different frequencies.
    """
    def __init__(self, input_dim=3, num_freqs=6):
        """
        Args:
            input_dim: Dimension of input coordinates (default: 3 for xyz)
            num_freqs: Number of frequency bands to use
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.output_dim = input_dim + input_dim * 2 * num_freqs
        self.freq_bands = 2.**torch.linspace(0., num_freqs - 1, num_freqs)

    def forward(self, x):
        """
        Args:
            x: Input coordinates, shape [..., input_dim]
            
        Returns:
            Encoded coordinates, shape [..., output_dim]
        """
        out = [x]
        self.freq_bands = self.freq_bands.to(x.device)
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)


class MiniPointNet(nn.Module):
    """
    Simplified PointNet for patch encoding.
    
    Processes a set of points with shared MLPs and max pooling
    to extract permutation-invariant features.
    """
    def __init__(self, input_dim=3, output_dim=256):
        """
        Args:
            input_dim: Dimension of input points (default: 3 for xyz)
            output_dim: Dimension of output features
        """
        super().__init__()
        self.mlp1 = nn.Conv1d(input_dim, 64, 1)
        self.mlp2 = nn.Conv1d(64, 128, 1)
        self.mlp3 = nn.Conv1d(128, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        """
        Args:
            x: Input points, shape [B, input_dim, N]
            
        Returns:
            Point features, shape [B, output_dim]
        """
        x = torch.relu(self.bn1(self.mlp1(x)))
        x = torch.relu(self.bn2(self.mlp2(x)))
        x = torch.relu(self.bn3(self.mlp3(x)))
        x = torch.max(x, dim=2)[0]  # Max pooling
        return x


class TransformerBlock(nn.Module):
    """
    Standard Transformer block with multi-head attention and feedforward network.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        """
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads
            ff_dim: Dimension of feedforward network
            dropout_rate: Dropout probability
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, 
            num_heads, 
            dropout=dropout_rate, 
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), 
            nn.GELU(), 
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        Args:
            x: Input embeddings, shape [B, N, C]
            attn_mask: Optional attention mask
            key_padding_mask: Optional key padding mask
            
        Returns:
            Processed embeddings, shape [B, N, C]
        """
        attn_output, _ = self.attn(
            x, x, x, 
            attn_mask=attn_mask, 
            key_padding_mask=key_padding_mask
        )
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pytorch3d.ops import knn_gather

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

def cos_angle(v1, v2):
    """
        V1, V2: (N, 3)
        return: (N,)
    """
    return torch.bmm(v1.unsqueeze(1), v2.unsqueeze(2)).view(-1) / torch.clamp(v1.norm(2, 1) * v2.norm(2, 1), min=0.000001)

