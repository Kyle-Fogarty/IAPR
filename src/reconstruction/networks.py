import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional dependency handling
try:
    import pytorch3d.ops
except ImportError:
    raise ImportError(
        "PyTorch3D is required for this implementation. "
        "Please install it using instructions from: "
        "https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md"
    )

def create_mlp(channels, batch_norm=True, use_conv2d=False):
    """
    Helper function to create MLPs with optional BatchNorm and ReLU.
    
    Args:
        channels (list[int]): List of channel dimensions for each layer.
        batch_norm (bool): Whether to use batch normalization. Default: True.
        use_conv2d (bool): Whether to use Conv2d instead of Conv1d. Default: False.
        
    Returns:
        nn.Sequential: Sequential module containing the MLP layers.
    """
    layers = []
    conv_layer = nn.Conv2d if use_conv2d else nn.Conv1d
    bn_layer = nn.BatchNorm2d if use_conv2d else nn.BatchNorm1d

    for i in range(len(channels) - 1):
        layers.append(conv_layer(channels[i], channels[i+1], kernel_size=1))
        if batch_norm:
            layers.append(bn_layer(channels[i+1]))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class PointNet2SetAbstraction(nn.Module):
    """
    PointNet++ Set Abstraction Module for downsampling points and extracting features.

    Args:
        npoint (int): Number of points to sample. Use None to group all points.
        radius (float): Radius for ball query grouping. Ignored if group_all=True.
        nsample (int): Max number of neighbors in ball query. Ignored if group_all=True.
        in_channels (int): Number of input feature channels (excluding xyz coordinates).
        mlp_channels (list[int]): List of channels for the MLP layers.
        group_all (bool): If True, group all points instead of sampling. Default: False.
    """
    def __init__(self, npoint, radius, nsample, in_channels, mlp_channels, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.input_feature_dim = in_channels

        # Define MLP with input as features + xyz (hence +3)
        mlp_spec = [in_channels + 3] + mlp_channels
        self.mlp_convs = create_mlp(mlp_spec, batch_norm=True, use_conv2d=True)

    def forward(self, xyz, features):
        """
        Forward pass of the Set Abstraction module.

        Args:
            xyz (Tensor): Input point coordinates (B, N, 3).
            features (Tensor or None): Input point features (B, C_in, N) or None.

        Returns:
            Tuple[Tensor, Tensor]:
                - new_xyz (Tensor): Sampled point coordinates (B, npoint or 1, 3).
                - new_features (Tensor): Features of sampled points (B, C_out, npoint or 1).
        """
        B, N, _ = xyz.shape
        
        # Validate feature dimensions if provided
        if features is not None:
            _B, C_in, _N = features.shape
            assert B == _B and N == _N and C_in == self.input_feature_dim, "Feature shape mismatch"
        else:
            assert self.input_feature_dim == 0, "Expected features but got None"
            C_in = 0

        # 1. Point Sampling - FPS or global pooling
        if self.group_all:
            # Global pooling case - use single centroid
            new_xyz = xyz.mean(dim=1, keepdim=True)  # (B, 1, 3)
            idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat(B, 1, 1)
            current_npoint, current_nsample = 1, N
        else:
            # Sample points using Farthest Point Sampling (FPS)
            current_npoint, current_nsample = self.npoint, self.nsample
            fps_idx = pytorch3d.ops.sample_farthest_points(xyz, K=current_npoint)[1]  # (B, npoint)
            new_xyz = torch.gather(xyz, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B, npoint, 3)
            
            # Find neighbors within radius using ball query
            idx = pytorch3d.ops.ball_query(
                p1=new_xyz, p2=xyz, K=current_nsample, 
                radius=self.radius, return_nn=False
            )[1]

        # 2. Group points and features
        idx_long = idx.long()
        B, P, K = idx_long.shape  # P = current_npoint, K = current_nsample

        # Group xyz coordinates
        grouped_xyz_list = []
        for b in range(B):
            batch_xyz = xyz[b]  # (N, 3)
            batch_idx = idx_long[b]  # (P, K)
            grouped_xyz_list.append(batch_xyz[batch_idx])
        grouped_xyz = torch.stack(grouped_xyz_list, dim=0)  # (B, P, K, 3)

        # Calculate local coordinates relative to centroids
        grouped_xyz_local = grouped_xyz - new_xyz.unsqueeze(2)  # (B, P, K, 3)

        # 3. Group features if provided
        if features is not None:
            features_perm = features.permute(0, 2, 1).contiguous()  # (B, N, C_in)
            grouped_features_list = []
            for b in range(B):
                batch_features = features_perm[b]  # (N, C_in)
                batch_idx = idx_long[b]  # (P, K)
                grouped_features_list.append(batch_features[batch_idx])
            grouped_features = torch.stack(grouped_features_list, dim=0)  # (B, P, K, C_in)
            
            # Concatenate local coordinates with gathered features
            grouped_point_features = torch.cat([grouped_xyz_local, grouped_features], dim=-1)
        else:
            # Use only local coordinates if no input features
            grouped_point_features = grouped_xyz_local

        # 4. Apply MLP to grouped features
        # Reshape for Conv2d: (B, P, K, C_in+3) -> (B, C_in+3, P, K)
        grouped_point_features = grouped_point_features.permute(0, 3, 1, 2).contiguous()
        
        # Apply MLP layers
        mlp_features = self.mlp_convs(grouped_point_features)  # (B, C_out, P, K)
        
        # 5. Max pooling over neighbors
        new_features = torch.max(mlp_features, dim=3)[0]  # (B, C_out, P)

        return new_xyz, new_features


class PointNet2FeaturePropagation(nn.Module):
    """
    PointNet++ Feature Propagation Module for upsampling features.

    Interpolates features from coarser points to finer points, 
    concatenates with skip connection features, and applies an MLP.

    Args:
        in_channels1 (int): Number of channels from the coarser level features.
        in_channels2 (int): Number of channels from the finer level skip features.
                            Can be 0 if no skip features exist.
        mlp_channels (list[int]): List of channels for the MLP layers.
        k (int): Number of nearest neighbors for interpolation. Default: 3.
    """
    def __init__(self, in_channels1, in_channels2, mlp_channels, k=3):
        super().__init__()
        self.k = k
        
        # MLP operates on concatenated features (interpolated + skip)
        mlp_spec = [in_channels1 + in_channels2] + mlp_channels
        self.mlp_convs = create_mlp(mlp_spec, batch_norm=True, use_conv2d=False)

    def forward(self, xyz1, xyz2, features1, features2):
        """
        Forward pass of the Feature Propagation module.

        Args:
            xyz1 (Tensor): Coarser level coordinates (B, N1, 3).
            xyz2 (Tensor): Finer level coordinates (B, N2, 3).
            features1 (Tensor): Coarser level features (B, C1, N1).
            features2 (Tensor or None): Finer level skip features (B, C2, N2) or None.

        Returns:
            Tensor: Propagated features for xyz2 points (B, C_out, N2).
        """
        B, N1, _ = xyz1.shape
        _, N2, _ = xyz2.shape
        _, C1, _ = features1.shape
        
        # Get C2 dimension from features2 if it exists
        if features2 is not None:
            _, C2, _ = features2.shape
        else:
            C2 = 0

        # 1. Find k nearest neighbors for interpolation
        dists, idx, _ = pytorch3d.ops.knn_points(xyz2, xyz1, K=self.k, return_nn=False)
        
        # 2. Calculate interpolation weights
        # Add epsilon to avoid division by zero
        dists = torch.clamp(dists, min=1e-10)
        inv_dists = 1.0 / dists  # (B, N2, k)
        
        # Normalize weights to sum to 1 for each point's neighbors
        weights = inv_dists / torch.sum(inv_dists, dim=2, keepdim=True)  # (B, N2, k)

        # 3. Interpolate features
        # Permute features1 to (B, N1, C1) for knn_gather
        features1_perm = features1.permute(0, 2, 1).contiguous()
        
        # Gather features of k neighbors
        interpolated_features = pytorch3d.ops.knn_gather(features1_perm, idx)  # (B, N2, k, C1)
        
        # Apply weights for interpolation
        interpolated_features = torch.sum(
            interpolated_features * weights.unsqueeze(-1), dim=2
        )  # (B, N2, C1)
        
        # Transpose back to (B, C1, N2) format
        interpolated_features = interpolated_features.permute(0, 2, 1).contiguous()

        # 4. Concatenate with skip features if available
        if features2 is not None:
            new_features = torch.cat([interpolated_features, features2], dim=1)  # (B, C1+C2, N2)
        else:
            new_features = interpolated_features  # (B, C1, N2)

        # 5. Apply MLP
        new_features = self.mlp_convs(new_features)  # (B, C_out, N2)

        return new_features


class PointNetPlusPlus(nn.Module):
    """
    PointNet++ network for extracting per-point features.
    
    Follows an encoder-decoder structure with skip connections.
    
    Args:
        input_feature_dim (int): Dimension of input features besides coordinates.
                                 Set to 0 if only coordinates are used as input.
        output_feature_dim (int): Dimension of output features per point. Default: 128.
        encoder_dims (list): Dimensions for encoder layers. Default is standard PointNet++ architecture.
        decoder_dims (list): Dimensions for decoder layers. Default is standard PointNet++ architecture.
    """
    def __init__(self, input_feature_dim=0, output_feature_dim=128, 
                 encoder_dims=None, decoder_dims=None):
        super().__init__()
        self.input_feature_dim = input_feature_dim
        self.output_feature_dim = output_feature_dim
        
        # Default encoder dimensions if not specified
        if encoder_dims is None:
            encoder_dims = [
                [64, 64, 128],     # SA1
                [128, 128, 256],   # SA2
                [256, 512, 1024]   # SA3 (global)
            ]
            
        # Default decoder dimensions if not specified
        if decoder_dims is None:
            decoder_dims = [
                [256, 256],        # FP3
                [256, 128],        # FP2
                [128, 128, output_feature_dim]  # FP1 (final output dimension)
            ]

        # Encoder (Set Abstraction Layers)
        self.sa1 = PointNet2SetAbstraction(
            npoint=512, radius=0.2, nsample=32,
            in_channels=input_feature_dim, mlp_channels=encoder_dims[0]
        )
        
        self.sa2 = PointNet2SetAbstraction(
            npoint=128, radius=0.4, nsample=64,
            in_channels=encoder_dims[0][-1], mlp_channels=encoder_dims[1]
        )
        
        # Global feature aggregation
        self.sa3 = PointNet2SetAbstraction(
            npoint=None, radius=None, nsample=None,
            in_channels=encoder_dims[1][-1], mlp_channels=encoder_dims[2], 
            group_all=True
        )

        # Decoder (Feature Propagation Layers)
        self.fp3 = PointNet2FeaturePropagation(
            in_channels1=encoder_dims[2][-1], in_channels2=encoder_dims[1][-1],
            mlp_channels=decoder_dims[0]
        )
        
        self.fp2 = PointNet2FeaturePropagation(
            in_channels1=decoder_dims[0][-1], in_channels2=encoder_dims[0][-1],
            mlp_channels=decoder_dims[1]
        )
        
        self.fp1 = PointNet2FeaturePropagation(
            in_channels1=decoder_dims[1][-1], in_channels2=input_feature_dim,
            mlp_channels=decoder_dims[2]
        )

    def forward(self, pointcloud):
        """
        Forward pass of the PointNet++ network.

        Args:
            pointcloud (Tensor): Input point cloud tensor (B, N, 3 + C_in).
                                 Assumes first 3 dimensions are XYZ coordinates.

        Returns:
            Tensor: Per-point features (B, C_out, N).
        """
        B, N, D = pointcloud.shape
        xyz = pointcloud[:, :, :3].contiguous()  # (B, N, 3)

        # Extract initial features if available
        if self.input_feature_dim > 0:
            features = pointcloud[:, :, 3:].transpose(1, 2).contiguous()  # (B, C_in, N)
        else:
            features = None

        # Encoder path
        l1_xyz, l1_features = self.sa1(xyz, features)              # (B, 512, 3), (B, 128, 512)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)        # (B, 128, 3), (B, 256, 128)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)        # (B, 1, 3), (B, 1024, 1)

        # Decoder path with skip connections
        fp3_features = self.fp3(l3_xyz, l2_xyz, l3_features, l2_features)  # (B, 256, 128)
        fp2_features = self.fp2(l2_xyz, l1_xyz, fp3_features, l1_features)  # (B, 128, 512)
        fp1_features = self.fp1(l1_xyz, xyz, fp2_features, features)        # (B, 128, N)

        return fp1_features.permute(0, 2, 1)


def test_pointnetpp():
    """Test function to verify the PointNet++ implementation with different output dimensions."""
    # Configuration
    batch_size = 2
    num_points = 1024
    input_coord_dim = 3
    input_feature_dims = 3  # Example: RGB as additional features
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dummy input: (B, N, 3+F)
    dummy_pointcloud = torch.randn(
        batch_size, num_points, input_coord_dim + input_feature_dims, 
        device=device
    )
    
    # Test with different output dimensions
    output_dims_to_test = [64, 128, 256]
    
    for output_dim in output_dims_to_test:
        print(f"\nTesting with output dimension: {output_dim}")
        
        # Initialize model with specified output dimension
        model = PointNetPlusPlus(
            input_feature_dim=input_feature_dims,
            output_feature_dim=output_dim
        ).to(device)
        
        # Forward pass
        try:
            with torch.no_grad():
                model.eval()
                output_features = model(dummy_pointcloud)
                
            print(f"Output features shape: {output_features.shape}")
            
            # Verify output shape
            expected_output_shape = (batch_size, output_dim, num_points)
            assert output_features.shape == expected_output_shape, \
                f"Output shape mismatch! Expected {expected_output_shape}, got {output_features.shape}"
                
            print(f"PointNet++ test with output_dim={output_dim} successful!")
            
            # Example: Get features in (B, N, C) format if needed
            output_features_per_point = output_features.permute(0, 2, 1)
            print(f"Output features per point shape: {output_features_per_point.shape}")
            
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # Test with custom encoder and decoder dimensions
    print("\nTesting with custom encoder and decoder dimensions")
    
    # Custom architecture configuration
    custom_encoder_dims = [
        [32, 64, 96],      # SA1: smaller than default
        [96, 128, 192],    # SA2: smaller than default
        [192, 384, 512]    # SA3: smaller than default
    ]
    
    custom_decoder_dims = [
        [256, 192],        # FP3: custom dimensions
        [192, 96],         # FP2: custom dimensions
        [96, 64, 32]       # FP1: custom output dimension of 32
    ]
    
    # Initialize model with custom dimensions
    custom_model = PointNetPlusPlus(
        input_feature_dim=input_feature_dims,
        output_feature_dim=32,  # This should match the last value in custom_decoder_dims[-1]
        encoder_dims=custom_encoder_dims,
        decoder_dims=custom_decoder_dims
    ).to(device)
    
    # Forward pass with custom model
    try:
        with torch.no_grad():
            custom_model.eval()
            custom_output = custom_model(dummy_pointcloud)
            
        print(f"Custom model output shape: {custom_output.shape}")
        
        # Verify output shape
        expected_custom_shape = (batch_size, 32, num_points)
        assert custom_output.shape == expected_custom_shape, \
            f"Custom output shape mismatch! Expected {expected_custom_shape}, got {custom_output.shape}"
            
        print("Custom PointNet++ test successful!")
        
    except Exception as e:
        print(f"Error during custom model forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True


if __name__ == '__main__':
    test_pointnetpp()