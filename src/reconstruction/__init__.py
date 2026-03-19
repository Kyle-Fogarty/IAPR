"""
RIMLS - Robust Implicit Moving Least Squares

A PyTorch implementation of the RIMLS algorithm for surface reconstruction
and processing of point clouds.
"""

# Basic PyTorch implementation
from .knn import find_knn_batched
from .rimls import RIMLS
from .utils import compute_average_spacing_from_knn
from .functional import (
    compute_potential,
    project_points,
    compute_potential_and_gradient,
    compute_weights_and_derivatives
)

# Check for Faiss availability
try:
    from .faiss_knn import FAISS_AVAILABLE, FAISS_GPU_AVAILABLE, FaissKNN, find_knn_faiss
    from .differentiable import DifferentiableRIMLSCore
    from .processor import RIMLSProcessor
    
    # Define what's available based on Faiss status
    if FAISS_AVAILABLE:
        __all__ = [
            # Basic implementation
            'find_knn_batched',
            'RIMLS',
            'compute_average_spacing_from_knn',
            'compute_potential',
            'project_points',
            'compute_potential_and_gradient',
            'compute_weights_and_derivatives',
            
            # Faiss-accelerated implementation
            'FAISS_AVAILABLE',
            'FAISS_GPU_AVAILABLE',
            'FaissKNN',
            'find_knn_faiss',
            'DifferentiableRIMLSCore',
            'RIMLSProcessor',
        ]
    else:
        __all__ = [
            'find_knn_batched',
            'RIMLS',
            'compute_average_spacing_from_knn',
            'compute_potential',
            'project_points',
            'compute_potential_and_gradient',
            'compute_weights_and_derivatives',
            'FAISS_AVAILABLE',
            'FAISS_GPU_AVAILABLE',
        ]
except ImportError:
    # Faiss not available, only expose basic implementation
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False
    
    __all__ = [
        'find_knn_batched',
        'RIMLS',
        'compute_average_spacing_from_knn',
        'compute_potential',
        'project_points',
        'compute_potential_and_gradient',
        'compute_weights_and_derivatives',
        'FAISS_AVAILABLE',
        'FAISS_GPU_AVAILABLE',
    ]

__version__ = '0.1.0'