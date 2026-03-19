"""
Accelerated KNN implementation using Faiss for RIMLS.
"""
import torch
import numpy as np
import time
from typing import Tuple, Optional, Dict, Any

# --- Faiss Import with Error Handling ---
try:
    import faiss
    FAISS_AVAILABLE = True
    
    # Check specifically for GPU Faiss availability
    try:
        res_test = faiss.StandardGpuResources()  # Test query GPU resources
        FAISS_GPU_AVAILABLE = faiss.get_num_gpus() > 0
        del res_test
    except AttributeError:
        # This might happen if faiss-cpu is installed but not faiss-gpu
        print("Warning: Faiss found, but GPU extensions (StandardGpuResources) seem missing. GPU Faiss KNN unavailable.")
        FAISS_GPU_AVAILABLE = False
    except Exception as e:
        print(f"Warning: Error checking Faiss GPU resources ({type(e).__name__}). GPU Faiss KNN unavailable.")
        FAISS_GPU_AVAILABLE = False

except ImportError:
    print("------------------------------------------------------------")
    print("WARNING: Faiss library not found or import failed!")
    print("For accelerated KNN, please install faiss-gpu for GPU support:")
    print("  conda install -c pytorch faiss-gpu=1.7.2 cudatoolkit=X.Y")
    print("  (Replace X.Y with your CUDA version, e.g., 11.3)")
    print("Or install faiss-cpu (GPU features will be disabled):")
    print("  pip install faiss-cpu")
    print("------------------------------------------------------------")
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False


class FaissKNN:
    """
    Fast K-Nearest Neighbors implementation using Faiss library.
    Handles both GPU and CPU implementations based on availability.
    """
    
    def __init__(self, 
                 device_preference: str = 'cuda', 
                 verbose: bool = False):
        """
        Initialize the Faiss KNN module.
        
        Args:
            device_preference: Preferred compute device ('cuda' or 'cpu')
            verbose: Whether to print status messages
        """
        if not FAISS_AVAILABLE:
            raise ImportError("Faiss library is required but not found. "
                              "Please install faiss-gpu or faiss-cpu.")
        
        self.device_preference = device_preference
        self.verbose = verbose
        self.device = None
        self._use_gpu_faiss = False
        self._faiss_index = None
        self._faiss_gpu_res = None
        self._indexed_vertices_ref = None
        
        self._setup_device()
    
    def _setup_device(self):
        """Determine the compute device and Faiss strategy based on availability."""
        if self.device_preference == 'cuda' and torch.cuda.is_available():
            if FAISS_GPU_AVAILABLE:
                self.device = torch.device("cuda")
                self._use_gpu_faiss = True
                if self.verbose: 
                    print(f"FaissKNN: Using device '{self.device}' with Faiss GPU.")
            else:
                self.device = torch.device("cuda")  # Use GPU for PyTorch tensors
                self._use_gpu_faiss = False  # But Faiss will run on CPU
                if self.verbose: 
                    print(f"FaissKNN: Using device '{self.device}'. "
                          "Faiss GPU not available, Faiss KNN will run on CPU.")
        else:
            self.device = torch.device("cpu")
            self._use_gpu_faiss = False
            if self.device_preference == 'cuda' and self.verbose:
                print("FaissKNN: CUDA specified but unavailable. Falling back to CPU.")
            if self.verbose: 
                print(f"FaissKNN: Using device '{self.device}' with Faiss CPU.")

    def build_index(self, 
                    vertices: torch.Tensor, 
                    force_rebuild: bool = False) -> None:
        """
        Build or rebuild the Faiss index for the given vertices.
        
        Args:
            vertices: Tensor (N, D) of vertices to index
            force_rebuild: If True, force rebuilding the index even if vertices seem unchanged
        """
        # Check if rebuild is needed
        if (not force_rebuild and 
            self._faiss_index is not None and 
            self._indexed_vertices_ref is vertices):
            if self.verbose: 
                print("FaissKNN: Faiss index already exists for the provided vertices. Skipping rebuild.")
            return
        
        if self.verbose: 
            print(f"FaissKNN: Building Faiss index...")
        start_time = time.time()
        
        # Ensure vertices are float32 and on CPU for Faiss indexing
        vertices_np = vertices.detach().to(dtype=torch.float32).cpu().numpy()
        
        if not np.all(np.isfinite(vertices_np)):
            raise ValueError("Vertices contain non-finite values (NaN/Inf). Cannot build Faiss index.")
        
        d = vertices_np.shape[1]  # Dimension (typically 3 for 3D points)
        self.release_resources()  # Clear any old index/resources first
        
        try:
            if self._use_gpu_faiss:
                if self.verbose: 
                    print("  Index Type: Faiss GPU (IndexFlatL2)")
                self._faiss_gpu_res = faiss.StandardGpuResources()
                gpu_device_id = self.device.index if self.device.index is not None else 0
                index_cpu = faiss.IndexFlatL2(d)
                self._faiss_index = faiss.index_cpu_to_gpu(self._faiss_gpu_res, gpu_device_id, index_cpu)
                self._faiss_index.add(vertices_np)
            else:  # Use CPU Faiss
                if self.verbose: 
                    print("  Index Type: Faiss CPU (IndexFlatL2)")
                self._faiss_index = faiss.IndexFlatL2(d)
                self._faiss_index.add(vertices_np)
            
            self._indexed_vertices_ref = vertices  # Store reference to track changes
            
            if self.verbose: 
                print(f"  Faiss index built in {time.time() - start_time:.3f} seconds.")
        
        except Exception as e:
            self.release_resources()  # Ensure cleanup on error
            print(f"\nERROR: Failed to build Faiss index: {type(e).__name__}: {e}")
            raise RuntimeError("Faiss index building failed.") from e
    
    def search(self, 
               query_points: torch.Tensor, 
               k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find k-nearest neighbors for query points.
        
        Args:
            query_points: Tensor (Q, D) of query points
            k: Number of nearest neighbors to find
            
        Returns:
            distances: Tensor (Q, k) of distances to k nearest neighbors
            indices: Tensor (Q, k) of indices of k nearest neighbors
        """
        if self._faiss_index is None:
            raise RuntimeError("Faiss index has not been built. Call build_index() first.")
        
        # Effective k depends on number of points in index
        n_index_points = self._faiss_index.ntotal
        effective_k = min(k, n_index_points)
        
        if effective_k < k and self.verbose:
            print(f"FaissKNN: Warning: effective k reduced to {effective_k} due to index size ({n_index_points}).")
        
        if effective_k == 0:  # Handle case where k=0 or index is empty
            q_count = query_points.shape[0]
            return (
                torch.zeros((q_count, 0), device=self.device),
                torch.zeros((q_count, 0), device=self.device, dtype=torch.long)
            )
        
        try:
            # Convert query points to NumPy for Faiss
            query_points_np = query_points.detach().cpu().numpy().astype(np.float32)
            
            # Perform KNN search
            distances_sq_np, indices_np = self._faiss_index.search(query_points_np, effective_k)
            
            # Handle potential negative distances (numerical issues)
            if np.any(distances_sq_np < -1e-6):
                if self.verbose:
                    print(f"Warning: Found {np.sum(distances_sq_np < -1e-6)} negative squared distances. "
                          f"Clamping to zero.")
                distances_sq_np = np.maximum(distances_sq_np, 0.0)
            
            # Convert back to PyTorch tensors
            distances = torch.sqrt(torch.from_numpy(distances_sq_np)).to(self.device)
            indices = torch.from_numpy(indices_np).to(self.device).long()
            
            # Safety check: clamp indices to valid range
            indices = torch.clamp(indices, 0, n_index_points - 1)
            
            return distances, indices
            
        except Exception as e:
            print(f"\nERROR: Faiss search failed: {type(e).__name__}: {e}")
            raise RuntimeError("Faiss KNN search failed.") from e
    
    def release_resources(self) -> None:
        """Explicitly release Faiss index and GPU resources."""
        if self._faiss_index is not None:
            if self.verbose: 
                print("FaissKNN: Releasing Faiss index.")
            del self._faiss_index
            self._faiss_index = None
        
        if self._faiss_gpu_res is not None:
            if self.verbose: 
                print("FaissKNN: Releasing Faiss GPU resources.")
            del self._faiss_gpu_res
            self._faiss_gpu_res = None
        
        self._indexed_vertices_ref = None
    
    def __del__(self):
        """Release resources when object is deleted."""
        self.release_resources()


def find_knn_faiss(query_points: torch.Tensor, 
                  vertices: torch.Tensor, 
                  k: int,
                  device: str = 'cuda',
                  verbose: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Functional interface to find k-nearest neighbors using Faiss.
    Creates a temporary FaissKNN object for one-time use.
    
    Args:
        query_points: Tensor (Q, D) of query points
        vertices: Tensor (N, D) of data points
        k: Number of neighbors to find
        device: Device to use ('cuda' or 'cpu')
        verbose: Whether to print status messages
        
    Returns:
        distances: Tensor (Q, k) of distances to k nearest neighbors
        indices: Tensor (Q, k) of indices of k nearest neighbors
    """
    if not FAISS_AVAILABLE:
        raise ImportError("Faiss library is required but not found.")
    
    faiss_knn = FaissKNN(device_preference=device, verbose=verbose)
    try:
        faiss_knn.build_index(vertices)
        distances, indices = faiss_knn.search(query_points, k)
        return distances, indices
    finally:
        faiss_knn.release_resources()