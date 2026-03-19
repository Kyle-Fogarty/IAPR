import lightning as L
import torch 
import torch.nn as nn


import faiss
import numpy as np
import torch
import torch.nn as nn


class LatentSetEncoder(nn.Module):
    def __init__(self, config, pointcloud):
        super().__init__()
        self.config = config
        
        self.pointcloud = pointcloud
        # (1) Build the KNN tree on the GPU.
        self.knn_tree = KNNTree(config, pointcloud) ## Can now be queried via self.knn_tree.query_knn(query_points, k)

        # (2)

        if config.sampling == 'static':
            print('Using Static Sampling')
            
        elif config.sampling == 'dynamic':
            print('Using Dynamic Sampling')
        else:
            raise ValueError(f'Invalid sampling method: {config.sampling}')
    def _perform_fps(self, num_points):
        
        return farthest_point_sampling(self.pointcloud, num_points)
    


    def forward(self,):

        # (1) If static sampling, then we can just
        #     use the KNN tree to get the latent set.


        return self.model(pointcloud)



class ImplicitAttentionField(L.LightningModule):
    def __init__(self, ):
        super().__init__()

        self.pointcloud2latentset = set_encoder()
        self.crossattn_field      = crossattn_field()




    def _encoder(self, pointcloud = None):
        
        if pointcloud is None:
            pointcloud = self.pointcloud
        

    def _decoder(self, latent_set, pointcloud):
        return self.model.decode(latent_set, pointcloud)


    def forward(self, query_points, pointcloud = None):
        if pointcloud is None:
            pointcloud = self.pointcloud

        # (1) Produce the latent set prior over the 
        #     pointcloud.
        
        latent_set = self._encoder(pointcloud)

        # (2) Query the attentive neural field at 
        #     the query points.

        scalar_field, gradient_field = self._decoder(latent_set, query_points)

        return scalar_field, gradient_field








class KNNTree(nn.Module):
    def __init__(self, config, pointcloud):
        super().__init__()
        self.config = config
        self.pointcloud = pointcloud
        
        # Initialize FAISS resources and index
        self._initialize_faiss_index()
        
        # Build the KNN tree
        self.build_knn_tree(pointcloud)
    
    def _initialize_faiss_index(self):
        """Initialize FAISS GPU resources and index for 3D points."""
        self.res        = faiss.StandardGpuResources()
        self.index      = faiss.IndexFlatL2(3)  # 3D points
        self.gpu_index  = faiss.index_cpu_to_gpu(self.res, 0, self.index)
    
    def _convert_to_numpy(self, tensor):
        """Convert PyTorch tensor to numpy array, detach and ensure CPU."""
        return tensor.detach().cpu().numpy().astype(np.float32)
    
    def _reshape_points(self, points):
        """Reshape point tensor to FAISS-compatible format."""
        return points.reshape(-1, 3)
    
    def build_knn_tree(self, pointcloud):
        """
        Build KNN tree for pointcloud using FAISS on GPU.
        
        Args:
            pointcloud: Input point cloud tensor [B, N, 3]
            
        Returns:
            index: FAISS GPU index
        """
        # Convert and reshape points for FAISS
        points_np = self._convert_to_numpy(pointcloud)
        points_np = self._reshape_points(points_np)
        
        # Add points to index
        self.gpu_index.add(points_np)
        
        return self.gpu_index
    
    def query_knn(self, query_points, k):
        """
        Query K nearest neighbors using FAISS.
        
        Args:
            query_points: Query points tensor [B, Q, 3]
            k: Number of neighbors to find
            
        Returns:
            distances: Distances to k neighbors [B, Q, k]
            indices: Indices of k neighbors [B, Q, k]
        """
        # Convert and reshape query points
        queries_np = self._convert_to_numpy(query_points)
        queries_np = self._reshape_points(queries_np)
        
        # Search for k nearest neighbors
        distances, indices = self.gpu_index.search(queries_np, k)
        
        # Reshape results back to batch format
        B = query_points.shape[0]
        Q = query_points.shape[1]
        
        distances = distances.reshape(B, Q, k)
        indices = indices.reshape(B, Q, k)
        
        return distances, indices
    
    def rebuild_tree(self, new_pointcloud):
        """
        Rebuild the KNN tree with new points.
        
        Args:
            new_pointcloud: New point cloud tensor [B, N, 3]
        """
        # Reset the index
        self.gpu_index.reset()
        self.pointcloud = new_pointcloud
        
        # Build new tree
        self.build_knn_tree(new_pointcloud)










