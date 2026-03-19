import argparse
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mode', type=str, default='')
    parser.add_argument('--log_root', type=str, default='./log')
    parser.add_argument('--data_set', type=str, default='',
                        choices=['PCPNet', 'FamousShape', 'FamousShape5k', 'SceneNN', 'Others', 'KITTI_sub', 'Semantic3D', '3DScene', 'WireframePC', 'NestPC', 'Plane'])
    ### Train
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
    parser.add_argument('--max_iter', type=int, default=20000)
    parser.add_argument('--save_inter', type=int, default=10000)
    parser.add_argument('--warn_up', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.0001)
    ### Dataset and loader
    parser.add_argument('--dataset_root', type=str, default='../NeuralGF/dataset/')
    parser.add_argument('--testset_list', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--num_points', type=int, default=5000)
    parser.add_argument('--num_query', type=int, default=10)
    parser.add_argument('--num_knn', type=int, default=64)
    parser.add_argument('--dis_k', type=int, default=64)
    parser.add_argument('--dis_scale', type=float, default=0.15)
    ### Test
    parser.add_argument('--ckpt_dir', type=str, default='')
    parser.add_argument('--ckpt_iter', type=int, default=None)
    parser.add_argument('--save_normal_npy', type=eval, default=False, choices=[True, False])
    parser.add_argument('--save_normal_xyz', type=eval, default=False, choices=[True, False])
    parser.add_argument('--save_mesh', type=eval, default=False, choices=[True, False])
    parser.add_argument('--avg_nor', type=eval, default=False, choices=[True, False])
    parser.add_argument('--mesh_far', type=float, default=-1.0)
    args = parser.parse_args()
    return args
