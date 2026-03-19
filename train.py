import sys
sys.path.append('.')

import os
import math
import argparse

import torch
import numpy as np
from tqdm import tqdm

from src.normal_est.data import PointCloudDataset
from src.model import ImplicitAttentionFields
from src.training_utils import save_normals_xyz, run_rimls, run_marching_cubes, run_field_cross_attn
from src.reconstruction import RIMLSProcessor


def parse_args():
    parser = argparse.ArgumentParser(description='Train Implicit Attention Prior for point cloud surface reconstruction')
    parser.add_argument('--input',       type=str,   default='inputs/virus_pca_normals_holes.xyz', help='Path to input point cloud (.xyz or .ply)')
    parser.add_argument('--output_dir',  type=str,   default='output',   help='Directory for checkpoints and results')
    parser.add_argument('--num_anchors', type=int,   default=4,          help='Number of dictionary anchor embeddings')
    parser.add_argument('--latent_dim',  type=int,   default=128,        help='Latent embedding dimension')
    parser.add_argument('--num_points',  type=int,   default=5000,       help='Points sampled per training iteration')
    parser.add_argument('--num_query',   type=int,   default=10,         help='Number of noisy query perturbations')
    parser.add_argument('--num_knn',     type=int,   default=64,         help='KNN neighbours for loss computation')
    parser.add_argument('--dis_k',       type=int,   default=64,         help='KNN neighbours for noise scale estimation')
    parser.add_argument('--dis_scale',   type=float, default=0.15,       help='Noise scale multiplier')
    parser.add_argument('--lr',          type=float, default=1e-4,       help='Learning rate')
    parser.add_argument('--max_iter',    type=int,   default=20000,      help='Total training iterations')
    parser.add_argument('--warn_up',     type=int,   default=10000,      help='Warm-up iterations')
    parser.add_argument('--save_every',  type=int,   default=5000,       help='Save checkpoint every N iterations')
    parser.add_argument('--eval_every',  type=int,   default=1000,       help='Export normals every N iterations')
    return parser.parse_args()


def update_learning_rate(optimizer, iter_step, init_lr, max_iter, warn_up):
    lr = (iter_step / warn_up) if iter_step < warn_up else \
         0.5 * (math.cos((iter_step - warn_up) / (max_iter - warn_up) * math.pi) + 1)
    lr = lr * init_lr
    for g in optimizer.param_groups:
        g['lr'] = lr


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = PointCloudDataset(
        args.input,
        num_points=args.num_points,
        num_query=args.num_query,
        num_knn=args.num_knn,
        dis_k=args.dis_k,
        dis_scale=args.dis_scale,
    )
    dataloader    = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True)
    iter_loader   = iter(dataloader)
    pcl_full      = torch.from_numpy(dataset.pointcloud).float().to(device).unsqueeze(0)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model     = ImplicitAttentionFields(num_anchor_points=args.num_anchors, latent_dim=args.latent_dim).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    processor = RIMLSProcessor()
    processor.build_index(torch.from_numpy(dataset.pointcloud).float())
    model.processor   = processor
    model.point_cloud = pcl_full

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_loss = float('inf')
    pbar      = tqdm(range(1, args.max_iter + 1))

    for step in pbar:
        update_learning_rate(optimizer, step, args.lr, args.max_iter, args.warn_up)

        data        = next(iter_loader)
        pcl_raw     = data['pcl_raw'].to(device)
        pcl_source  = data['pcl_source'].to(device)
        knn_idx     = data['knn_idx'].to(device)
        pcl_raw_sub = data['pcl_raw_sub'].to(device)

        model.train()
        optimizer.zero_grad()

        pcl_source = torch.cat([pcl_source, pcl_raw_sub], dim=-2)

        model.run_forward(pcl_full, pcl_source)
        loss, _ = model.get_loss(pcl_raw=pcl_raw, pcl_source=pcl_source, knn_idx=knn_idx)

        loss.backward()
        optimizer.step()

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(loss=f'{loss.item():.6f}', lr=f'{current_lr:.6f}')

        # Save best checkpoint
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({'iteration': step, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': best_loss},
                       os.path.join(args.output_dir, 'best_model.pth'))

        # Periodic checkpoint
        if step % args.save_every == 0:
            torch.save({'iteration': step, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), 'loss': loss.item()},
                       os.path.join(args.output_dir, f'model_iter_{step}.pth'))

        # Export intermediate normals
        if step % args.eval_every == 0:
            pcl_pts = torch.from_numpy(dataset.pointcloud).float().to(device)
            save_normals_xyz(model, pcl_pts, args.output_dir, f'normals_iter_{step}')

    # ------------------------------------------------------------------
    # Final extraction
    # ------------------------------------------------------------------
    print('\nExtracting final normals and surface mesh...')
    model.weight   = 0.5
    pcl_pts        = torch.from_numpy(dataset.pointcloud).float().to(device)
    pred_normals   = save_normals_xyz(model, pcl_pts, args.output_dir, 'normals_final', cal_attention_weights=True)

    sdf_grid, _    = run_rimls(processor, pcl_pts, pred_normals)
    run_marching_cubes(sdf_grid, pcl_pts, pred_normals, args.output_dir, 'mesh_final')

    sdf_attn, _    = run_field_cross_attn(model, pcl_pts, cal_attention_weights=True)
    run_marching_cubes(sdf_attn, pcl_pts, pred_normals, args.output_dir, 'mesh_attn_field')

    print(f'\nDone. Results saved to: {args.output_dir}/')


if __name__ == '__main__':
    main()
