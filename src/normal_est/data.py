

import os
import sys
import time
import numpy as np
import scipy.spatial as spatial
import torch
from torch.utils.data import Dataset
import trimesh 

def load_data(filedir, filename, dtype=np.float32, wo=False):
    d = None
    filepath = os.path.join(filedir, 'npy', filename + '.npy')
    os.makedirs(os.path.join(filedir, 'npy'), exist_ok=True)
    if os.path.exists(filepath):
        if wo:
            return True
        d = np.load(filepath)
    else:
        d = np.loadtxt(os.path.join(filedir, filename), dtype=dtype)
        np.save(filepath, d)
    return d


def normalization(pcl):
    """
        pcl: (N, 3)
    """
    shape_scale = np.max([np.max(pcl[:,0])-np.min(pcl[:,0]), np.max(pcl[:,1])-np.min(pcl[:,1]), np.max(pcl[:,2])-np.min(pcl[:,2])])
    shape_center = [(np.max(pcl[:,0])+np.min(pcl[:,0]))/2, (np.max(pcl[:,1])+np.min(pcl[:,1]))/2, (np.max(pcl[:,2])+np.min(pcl[:,2]))/2]
    pcl = pcl - shape_center
    pcl = pcl / shape_scale
    return pcl



import trimesh 
from tqdm import tqdm

class PointCloudDataset(Dataset):
    def __init__(self, root, num_points=5000, num_query=10, num_knn=64, dis_k=50, dis_scale=1.0):
        super().__init__()


        self.pointcloud_path = root
        self.pointcloud      = self.load_pointcloud()
        self.pcl_full = torch.from_numpy(self.pointcloud).float().unsqueeze(0)
        self.num_points = num_points
        self.num_query  = num_query
        self.num_knn    = num_knn
        self.dis_k      = dis_k
        self.dis_scale = dis_scale
        self.num_split = 10
        self.max_point = int(3e5)

        self.process_data()



    def load_pointcloud(self):

        print(f'Loading Pointcloud: @ {self.pointcloud_path}')
        try:
            pointcloud = np.array(trimesh.load(self.pointcloud_path).vertices)
        except Exception as e:
            print(f"Error loading point cloud from {self.pointcloud_path}: {str(e)}")
            raise
        print(f'Pointcloud contains: ({pointcloud.shape[0]}) points')
        pointcloud = normalization(pointcloud)
        return pointcloud

    

    def process_data(self):
        self.pcl_raw = None
        self.k_idex = None
        self.pt_source = None
        self.knn_idx = None

        start_time = time.time()
        pointcloud, normal_gt = self.pointcloud, self.pointcloud#self.get_data(shape_name)

        ## Do some subsampling and find the number of queries.

        if pointcloud.shape[0] > self.max_point:
            print('Using sparse point cloud data: %d' % self.max_point)
            pidx = np.random.choice(pointcloud.shape[0], self.max_point, replace=False)
            pointcloud = pointcloud[pidx, :]

        if 1000000 / pointcloud.shape[0] <= 10.0:
            num_query = self.num_query
        else:
            num_query = 1000000 // pointcloud.shape[0] 




        sigmas = []
        k_idex = []
        ptree = spatial.cKDTree(pointcloud)
        
        
        for p in np.array_split(pointcloud, 100, axis=0):
            d, idex = ptree.query(p, k=self.dis_k + 1)  # no self
            # d = np.clip(d, a_min=0, a_max=0.5)
            sigmas.append(d[:, -1])                                         # Pull the 50th distance 
            k_idex.append(idex)
        sigmas = np.concatenate(sigmas, axis=0)[:, None]     # (N, 1)       # The distance to the 50th NN
        self.k_idex = np.concatenate(k_idex, axis=0)         # (N, K)       # Index list for mapping point -> 51 NN.
        # sigmas[sigmas > 2 * sigmas.mean()] = 2 * sigmas.mean()    

        sample  = []
        knn_idx = []


        ## ============[Workout the distance sales]============

        if self.dis_scale == 1.0 or self.dis_scale * np.sqrt(pointcloud.shape[0] / 20000) < self.dis_scale:
            dis_scale = self.dis_scale
        else:
            dis_scale = self.dis_scale * np.sqrt(pointcloud.shape[0] / 20000)



        ## ================[Generate Query Points and NN]================
        print(f"[CONSTRUCTING NOISED DATASET]")
        for i in tqdm(range(num_query)):
            
            ## p' = p + N(0, 1) * dist(50 NN) * dist_scale[default 0.15]
            pcl_noisy = pointcloud + np.random.normal(0.0, 1.0, size=pointcloud.shape) * sigmas * dis_scale
            

            sample.append(pcl_noisy)

            # For each noisy query points, find the 64 NN
            for p in np.array_split(pcl_noisy, 100, axis=0):
                _, index = ptree.query(p, k=self.num_knn)
                knn_idx.append(index)
            # print(i, 'Processing', shape_name)

        self.pt_source  = np.concatenate(sample, axis=0)        # noisy point cloud, (N * num_query, 3)
        self.knn_idx    = np.concatenate(knn_idx, axis=0)       # (N * num_query, K)
        

        if self.num_knn == 1:
            self.knn_idx = self.knn_idx[:, None]

        self.pt_num = self.pt_source.shape[0] - 1
        elapsed_time = time.time() - start_time              # time second

        self.pcl_raw = torch.from_numpy(pointcloud).float()  # (N, 3)
        self.k_idex = torch.from_numpy(self.k_idex).long()   # (N, K1)
        # print(shape_name, 'Size:', self.pt_source.shape, '| Time: %.3f sec' % elapsed_time, '\n')

    def __len__(self):
        return self.pt_source.shape[0]

    def __getitem__(self, idx):
        
        index_coarse = np.random.choice(self.num_split, 1)
        index_fine   = np.random.choice(self.pt_num//self.num_split, self.num_points, replace=False)
        index        = index_fine * self.num_split + index_coarse

        pidx = np.random.choice(self.pcl_raw.shape[0], self.num_points//2, replace=False)
        pcl_raw_sub = self.pcl_raw[pidx]

        # knn_idx_sub = self.knn_idx[index, 0:1]
        # pcl_raw_sub = knn_gather_np(self.pointcloud, knn_idx_sub)[:,0,:]
        # pcl_raw_sub = torch.from_numpy(pcl_raw_sub).float()

        data = {
            'pcl_raw': self.pcl_raw,
            # 'k_idex': self.k_idex[pidx],
            'pcl_raw_sub': pcl_raw_sub,
            'pcl_source': torch.from_numpy(self.pt_source[index]).float(),
            'knn_idx': torch.from_numpy(self.knn_idx[index]).long(),
        }
        return data
