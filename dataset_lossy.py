import numpy as np
import torch
import torch.utils.data as data
from os.path import join
import os
import MinkowskiEngine as ME
import random
import open3d


class Dataset(data.Dataset):
    def __init__(self, root_dir, split, bit=10, maximum=20475, type='train', scaling_factor=1, time_step=1, return_normal=False):
        self.maximum = maximum
        self.type = type
        self.scaling_factor = scaling_factor
        self.return_normal = return_normal
        sequence_list = ['soldier', 'redandblack', 'loot', 'longdress', 'andrew', 'basketballplayer', 'dancer', 'david', 'exercise', 'phil', 'queen', 'ricardo', 'sarah', 'model']
        self.sequence_list = sequence_list
        start = [536, 1450, 1000, 1051, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1]
        end = [835, 1749, 1299, 1350, 317, 600, 600, 215, 600, 244, 249, 215, 206, 600]
        num = [end[i] - start[i] for i in range(len(start))]
        self.lookup = []
        for i in split:
            sequence_dir = join(root_dir, sequence_list[i]+'_ori')
            # sequence_dir = join(root_dir, sequence_list[i])
            file_prefix = sequence_list[i]+'_vox'+str(bit)+'_'
            file_subfix = '.npy'
            if type == 'train':
                s = start[i]
                e = int((end[i]-start[i])*0.95+start[i])
            elif type == 'val':
                s = int((end[i]-start[i])*0.95 +start[i])
                e = end[i]-time_step+1
            else:
                s = start[i]
                e = end[i]
            for s in range(s, e):
                s1 = str(s+time_step).zfill(4)
                s0 = str(s).zfill(4)
                file_name0 = file_prefix + s0 + file_subfix
                file_name1 = file_prefix + s1 + file_subfix
                file_dir = join(sequence_dir, file_name0)
                file_dir1 = join(sequence_dir, file_name1)
                self.lookup.append([file_dir, file_dir1])

    def __getitem__(self, item):
        file_dir, file_dir1 = self.lookup[item]
        p, p1 = np.load(file_dir), np.load(file_dir1)
        pc = torch.tensor(p[:, :3]).cuda()
        pc1 = torch.tensor(p1[:, :3]).cuda()
        if self.type == 'train':
            pc, pc1 = random_crop(pc, pc1, 1024, 700)
            # write_ply_data('f1.ply', pc.cpu().numpy())
            # write_ply_data('f2.ply', pc1.cpu().numpy())

        if self.scaling_factor != 1:
            pc = torch.unique(torch.floor(pc / self.scaling_factor[0] * self.scaling_factor[1]), dim=0)
            pc1 = torch.unique(torch.floor(pc1 / self.scaling_factor[0] * self.scaling_factor[1]), dim=0)
        xyz, point = pc, torch.ones_like(pc[:, :1])
        xyz1, point1 = pc1, torch.ones_like(pc1[:, :1])

        if not self.return_normal:
            return xyz, point, xyz1, point1
        else:
            n = torch.tensor(p[:, 4:7]).cuda()
            n1 = torch.tensor(p1[:, 4:7]).cuda()
            return xyz, point, n, xyz1, point1, n1

    def __len__(self):
        return len(self.lookup)


def random_crop(pc1, pc2, maximum, size):
    while True:
        # print(x,y,z,x_,y_,z_)
        x, y, z = random.randint(0, maximum - size - 1), random.randint(0, maximum - size - 1), \
                  random.randint(0, maximum - size - 1)
        x_, y_, z_ = x + size, y + size, z + size
        pc2_ = pc2[(x<=pc2[:, 0]) & (pc2[:, 0]<=x_) & (y<=pc2[:, 1]) & (pc2[:, 1]<=y_) & (z<=pc2[:, 2]) & (pc2[:, 2]<=z_)]
        pc1_ = pc1[(x-64 <= pc1[:, 0]) & (pc1[:, 0]<= x_+64) & (y-64 <= pc1[:, 1]) & (pc1[:, 1]<= y_+64) & (z-64 <= pc1[:, 2]) & (pc1[:, 2]<= z_+64)]
        # print(pc2_.size(), pc1_.size())
        if 600000 >= pc2_.shape[0] >= 100000:
            break
    return pc1_, pc2_


def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError('No data in the batch')

    # coords, feats, labels = list(zip(*list_data))
    # print(len(list(zip(*list_data))), "?")
    xyz, point, xyz1, point1 = list(zip(*list_data))

    xyz_batch = ME.utils.batched_coordinates(xyz)
    point_batch = torch.vstack(point).float()
    xyz1_batch = ME.utils.batched_coordinates(xyz1)
    point1_batch = torch.vstack(point1).float()
    return xyz_batch, point_batch, xyz1_batch, point1_batch


def collate_pointcloud_fn_normal(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError('No data in the batch')

    # coords, feats, labels = list(zip(*list_data))
    xyz, point, n, xyz1, point1, n1 = list(zip(*list_data))

    xyz_batch = ME.utils.batched_coordinates(xyz)
    point_batch = torch.vstack(point).float()
    n_batch = torch.vstack(n)
    xyz1_batch = ME.utils.batched_coordinates(xyz1)
    point1_batch = torch.vstack(point1).float()
    n1_batch = torch.vstack(n1)
    return xyz_batch, point_batch, n_batch, xyz1_batch, point1_batch, n1_batch


def write_ply_data(filename, points):
    if os.path.exists(filename):
        os.system('rm ' + filename)
    f = open(filename, 'a+')
    # print('data.shape:',data.shape)
    f.writelines(['ply\n', 'format ascii 1.0\n'])
    f.write('element vertex ' + str(points.shape[0]) + '\n')
    f.writelines(['property float x\n', 'property float y\n', 'property float z\n'])
    f.write('end_header\n')
    for _, point in enumerate(points):
        f.writelines([str(point[0]), ' ', str(point[1]), ' ', str(point[2]), '\n'])
    f.close()
    return


if __name__ == '__main__':
    d = Dataset(root_dir = '/home/zhaoxudong/dataset_npy', split=[0,1,2,3], type='train')
    print(d[10][2].shape)