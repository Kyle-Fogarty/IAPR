import torch 
# --- Configuration ---

class Config:
    def __init__(self) -> None:
        self.config = {
    "mesh_path": "input/hand.ply",
    "batch_size": 1, # Usually 1 for shape reconstruction
    "num_input_points": 10000, # Points sampled from mesh surface as input
    "num_query_points_train": 4096, # Points sampled in space for SDF supervision during training
    "model": {
        "m_anchors": 1024,
        "k_neighbors": 64,
        "latent_dim": 32,
        "num_enc_blocks": 4,
    },
    "training": {
        "num_epochs": 10000, # Reduced for quicker testing, adjust as needed
        "learning_rate": 1e-4,
        "loss_lambda_sdf": 1.0,
        "print_interval": 10, # Print loss every N epochs
        "checkpoint_dir": "output/checkpoints_hand",
        "best_model_path": "output/checkpoints_hand/best_model.pth",
    },
    "mesh_extraction": {
        "resolution": 128, # Grid resolution for marching cubes
        "batch_points": 32768, # Points to query in one go during extraction
        "output_path": "extracted_mesh_hand.ply",
        "padding": 0.1 # Padding around the mesh bounds for the grid
    },
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}