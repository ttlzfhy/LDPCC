import open3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np
from pytorch3d.ops import knn_points
import os
import time

from models.resnet import ResNet, InceptionResNet
import math


def masked_softmax(A, mask, dim):
    A_max = torch.max(A, dim=dim, keepdim=True)[0]
    A_exp = torch.exp(A - A_max)
    A_exp = A_exp * mask
    A_softmax = A_exp / (torch.sum(A_exp, dim=dim, keepdim=True) + 1e-8)
    return A_softmax


def ball_knn(xyz1, xyz2, K, radius):
    """
    K nearest neighbors in xyz2 for each points in xyz1
    """
    dist, idx, _ = knn_points(xyz1, xyz2, K=K, return_nn=False)
    mask = dist <= radius
    return dist, idx, mask


def sort_by_coor_sum(f, stride=None):
    if stride is None:
        stride = f.tensor_stride[0]
    xyz = f.C.clone()
    maximum = xyz.max() + 1
    xyz, maximum = xyz.long(), maximum.long()
    # non-negative (positive values only for sort)
    if xyz.min() < 0:
        min_value = xyz.min()
        xyz[:, 1:] = xyz[:, 1:] - min_value
        maximum = maximum - min_value
    assert xyz.min() >= 0 and xyz.max() - xyz.min() < maximum
    # coord to 1D
    coor_sum = xyz[:, 0] * maximum * maximum * maximum \
               + xyz[:, 1] * maximum * maximum \
               + xyz[:, 2] * maximum \
               + xyz[:, 3]
    # sort
    _, idx = coor_sum.sort()
    xyz_, feature_ = f.C[idx], f.F[idx]
    f_ = ME.SparseTensor(feature_, coordinates=xyz_, tensor_stride=stride, device=f.device)

    return f_


def coordinate_sort_by_coor_sum(xyz_in):
    maximum = xyz_in.max() + 1
    xyz, maximum = xyz_in.clone().long(), maximum.long()
    # non-negative (positive values only for sort)
    if xyz.min() < 0:
        min_value = xyz.min()
        xyz[:, 1:] = xyz[:, 1:] - min_value
        maximum = maximum - min_value
    assert xyz.min() >= 0 and xyz.max() - xyz.min() < maximum
    # coord to 1D
    coor_sum = xyz[:, 0] * maximum * maximum * maximum \
               + xyz[:, 1] * maximum * maximum \
               + xyz[:, 2] * maximum \
               + xyz[:, 3]
    _, idx = coor_sum.sort()
    xyz_ = xyz_in[idx].to(torch.float32)
    return xyz_


def index_points(points, idx):
    """
    Input:
        points: input points data, [B*C, N1, 1]
        idx: sample index data, [B*C, N2, K]
    Return:
        new_points:, indexed points data, [B*C, N2, K, 1]
    """
    device = points.device
    B = points.shape[0]  # B*C
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)  # B*C,1,1
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1  # 1,N2,K
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    # B*C,1,1 -> B*C,N2,K
    new_points = points[batch_indices, idx, :]
    return new_points


def quant(x, training=False, qs=1):
    if training:
        compressed_x = x + torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5) * qs
    else:
        compressed_x = torch.round(x / qs) * qs
    return compressed_x


def merge_two_frames(f1, f2):
    stride = f1.tensor_stride[0]
    f1_ = ME.SparseTensor(torch.cat([f1.F, torch.zeros_like(f1.F)], dim=-1), coordinates=f1.C,
                          tensor_stride=stride, device=f1.device)
    f2_ = ME.SparseTensor(torch.cat([torch.zeros_like(f2.F), f2.F], dim=-1), coordinates=f2.C,
                          tensor_stride=stride, coordinate_manager=f1_.coordinate_manager, device=f1.device)
    merged_f = f1_ + f2_
    merged_f = ME.SparseTensor(merged_f.F, coordinates=merged_f.C, tensor_stride=stride, device=merged_f.device)
    return merged_f


def index_by_channel(point1, idx, K=3):
    B, N1, C = point1.size()
    _, N2, C, __ = idx.size()  # (B, N2, C, K)
    point1_ = point1.transpose(1, 2).reshape(-1, N1, 1)  # (B*C, N1, 1)
    idx_ = idx.transpose(1, 2).reshape(-1, N2, K)  # (B*C, N2, K)
    knn_point1 = index_points(point1_, idx_).reshape(B, C, N2, K).transpose(1, 2)
    # knn_point1 = point1_[np.arange(B * C), idx_].transpose(1, 2)
    return knn_point1


def get_target_by_sp_tensor(out, target_sp_tensor):
    with torch.no_grad():
        def ravel_multi_index(coords, step):
            coords = coords.long()
            step = step.long()
            coords_sum = coords[:, 3] \
                         + coords[:, 2] * step \
                         + coords[:, 1] * step * step \
                         + coords[:, 0] * step * step * step
            return coords_sum

        # non-negative
        out_coords, target_coords = out.C.clone(), target_sp_tensor.C.clone()
        min_value = torch.min(out_coords.min(), target_coords.min())  # must the same min_value!!!
        if min_value < 0:
            out_coords[:, 1:] -= min_value
            target_coords[:, 1:] -= min_value

        step = max(out_coords.max(), target_coords.max()) + 1

        out_sp_tensor_coords_1d = ravel_multi_index(out_coords, step)
        in_sp_tensor_coords_1d = ravel_multi_index(target_coords, step)

        # test whether each element of a 1-D array is also present in a second array.
        target = torch.isin(out_sp_tensor_coords_1d, in_sp_tensor_coords_1d)

    return target


def get_coords_nums_by_key(out, target_key):
    with torch.no_grad():
        cm = out.coordinate_manager
        strided_target_key = cm.stride(target_key, out.tensor_stride[0])

        ins = cm.get_kernel_map(
            out.coordinate_map_key,
            strided_target_key,
            kernel_size=1,
            region_type=1)

        row_indices_per_batch = out._batchwise_row_indices
        # print(ins)
        # print(row_indices_per_batch)

        coords_nums = [len(np.in1d(row_indices, ins[0]).nonzero()[0]) for _, row_indices in
                       enumerate(row_indices_per_batch)]
        # coords_nums = [len(np.intersect1d(row_indices,ins[0])) for _, row_indices in enumerate(row_indices_per_batch)]

    return coords_nums


def keep_adaptive(out, coords_nums, rho=1.0):
    with torch.no_grad():
        keep = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        #  get row indices per batch.
        # row_indices_per_batch = out.coords_man.get_row_indices_per_batch(out.coordinate_map_key)
        row_indices_per_batch = out._batchwise_row_indices

        for row_indices, ori_coords_num in zip(row_indices_per_batch, coords_nums):
            coords_num = min(len(row_indices), ori_coords_num * rho)  # select top k points.
            values, indices = torch.topk(out.F[row_indices].squeeze(), int(coords_num))
            keep[row_indices[indices]] = True
    return keep


class DynamicFilter(nn.Module):
    def __init__(self, in_channel, mlp, residual=True, downsample_rate=1, last_layer=False):
        super(DynamicFilter, self).__init__()
        self.last_layer = last_layer
        self.dr = downsample_rate
        # self.depth_kernel_mlp = nn.Sequential(
        #     nn.Linear(4, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, in_channel + 4)
        # )
        self.depth_kernel_mlp = nn.Sequential(
            nn.Linear(4 + in_channel, 32),
            nn.ReLU(),
            nn.Linear(32, in_channel + 4)
        )
        self.mlp = nn.Sequential()  # nn.Linear(132/68, 64) -- ReLU -- nn.Linear(64, 64)
        start = in_channel + 4
        for i, hidden_size in enumerate(mlp):  # mlp=[]
            self.mlp.add_module('fc' + str(i), nn.Linear(start, hidden_size))  # 叠加2层
            if i != len(mlp) - 1:
                self.mlp.add_module('ReLu' + str(i), nn.ReLU(inplace=True))
            start = hidden_size

        self.residual = residual
        if residual:
            self.shortcut = nn.Linear(in_channel, mlp[-1])

    def forward(self, xyz, xyz_nn, points, knn, dists, mask=None, get_sum=True):
        """
        Args:
            xyz: input coordinates
            xyz_nn: xyz_nn contains K nearest neighbors of each point in xyz
            points: input features
            knn: knn_idx
            dists: sqrt(knn_dist)
            mask: mask of knn neighbors in the ball
            get_sum: if True, output feature is weighted-average sum

        Intermediate:
            grouped_xyz: KNN邻居相对坐标+距离开方, channels=4
            grouped_points: KNN邻居相对坐标+距离开方+特征, channels=4+128/64 (4+in_channels)
            dynamic_kernel: 加权的权重, 由grouped_xyz生成

        Returns:
            sampled_xyz: 即input xyz
            grouped_points: 加权后的特征
        """
        B, N, _ = xyz.size()
        if mask is None:
            mask = torch.ones_like(knn)
        else:
            mask = mask.to(torch.float32)
        sampled_xyz, sampled_knn, sampled_dists = xyz, knn, dists
        sampled_xyz_nn = index_points(xyz_nn, sampled_knn)
        grouped_xyz = torch.cat([sampled_xyz_nn - sampled_xyz.unsqueeze(2), sampled_dists.unsqueeze(3)], dim=-1)
        if points is None:
            grouped_points = grouped_xyz
        elif len(list(points.size())) == 3:
            grouped_points = torch.cat([grouped_xyz, index_points(points, sampled_knn)], dim=-1)
        else:
            grouped_points = torch.cat([grouped_xyz, points], dim=-1)
        # dynamic_kernel = self.depth_kernel_mlp(grouped_xyz)
        dynamic_kernel = self.depth_kernel_mlp(grouped_points)
        dynamic_kernel = masked_softmax(dynamic_kernel, mask.unsqueeze(-1), 2)
        if get_sum:
            grouped_points = torch.sum(dynamic_kernel * grouped_points, dim=2)
        else:
            grouped_points = torch.max(dynamic_kernel * grouped_points, dim=2)
        # grouped_points = torch.cat([sampled_xyz, grouped_points], dim=-1)
        grouped_points = self.mlp(grouped_points)
        if self.residual:
            return sampled_xyz, self.shortcut(points) + grouped_points
        else:
            return sampled_xyz, grouped_points


class inter_prediction(nn.Module):
    def __init__(self, input, hidden=64, output=8, kernel_size=2):
        super(inter_prediction, self).__init__()
        # self.conv1 = ME.MinkowskiConvolution(in_channels=input + input, out_channels=hidden, kernel_size=3, stride=1,
        #                                      bias=True,
        #                                      dimension=3)
        # self.conv2 = ME.MinkowskiConvolution(in_channels=hidden, out_channels=hidden, kernel_size=3, stride=1,
        #                                      bias=True,
        #                                      dimension=3)
        self.conv1 = DynamicFilter(input + input, [hidden, hidden], residual=False)
        self.conv2 = DynamicFilter(hidden, [hidden, hidden], residual=True)
        self.down1 = DownsampleWithPruning(hidden, hidden, 3, 2, ResNet)
        self.up2 = DeconvWithPruning(hidden, hidden)
        self.down2 = DownsampleWithPruning(hidden, hidden, 3, 2, None)

        self.motion_compressor = ME.MinkowskiConvolution(in_channels=hidden, out_channels=output, kernel_size=2,
                                                         stride=2,
                                                         bias=True,
                                                         dimension=3)
        self.motion_decompressor1 = DeconvWithPruning(output, hidden)
        '''
        self.motion_compressor = ME.MinkowskiConvolution(in_channels=hidden, out_channels=output, kernel_size=3,
                                                         stride=1,
                                                         bias=True,
                                                         dimension=3)
        self.motion_decompressor1 = ME.MinkowskiGenerativeConvolutionTranspose(in_channels=output, out_channels=hidden,
                                                                               kernel_size=3,
                                                                               stride=1,
                                                                               bias=True,
                                                                               dimension=3)
        '''
        self.motion_decompressor2 = DeconvWithPruning(hidden, hidden)
        self.high_resolution_motion_generator = ME.MinkowskiConvolution(in_channels=hidden, out_channels=input * 3,
                                                                        kernel_size=3, stride=1,
                                                                        bias=True,
                                                                        dimension=3)
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.unpooling = ME.MinkowskiPoolingTranspose(kernel_size=2, stride=2, dimension=3)
        self.low_resolution_motion_generator = ME.MinkowskiConvolution(in_channels=hidden, out_channels=input * 3,
                                                                       kernel_size=3, stride=1,
                                                                       bias=True,
                                                                       dimension=3)

        # reference convolution, to find the coordinate
        self.conv_ref = ME.MinkowskiConvolution(in_channels=input, out_channels=1, kernel_size=2, stride=2,
                                                bias=True,
                                                dimension=3)
        self.conv_ref2 = ME.MinkowskiConvolution(in_channels=1, out_channels=1, kernel_size=2, stride=2,
                                                 bias=True,
                                                 dimension=3)
        self.pruning = ME.MinkowskiPruning()
        self.relu_point = nn.ReLU(inplace=True)

    def prune(self, f1, f2):
        mask = get_target_by_sp_tensor(f1, f2)
        out = self.pruning(f1, mask.to(f1.device))
        return out

    def get_downsampled_coordinate(self, x, stride, return_sorted=False, return_sparse=False):
        pc = ME.SparseTensor(torch.ones([x.size()[0], 1], dtype=torch.float32, device=x.device), coordinates=x, tensor_stride=stride)
        downsampled = self.conv_ref2(pc)
        if return_sorted:
            downsampled = sort_by_coor_sum(downsampled, stride)
        if return_sparse:
            return downsampled
        return downsampled.C

    def get_motion_vector_1(self, f1, f2, stride=8, K=16, r=3):
        print('get_motion_vector_1 enable')
        # motion estimation
        merged_f = merge_two_frames(f1, f2)
        # out1 = self.relu(self.conv1(merged_f))
        # e_o = self.relu(self.conv2(out1))

        xyz, feature = merged_f.C, merged_f.F
        xyz, feature = (xyz.to(torch.float32)/stride).unsqueeze(0)[:, :, 1:], feature.unsqueeze(0)  # 球KNN去除球面外的邻居，坐标需要先除以步长
        dist, knn_xyz, mask = ball_knn(xyz, xyz, K=16, radius=3)
        xyz = xyz/5  # mlp特征提取中涉及坐标和距离的concate，需要减小坐标以避免坐标值与距离数量级相差太大
        out1 = self.relu_point(self.conv1(xyz, xyz, feature, knn_xyz, dist.sqrt(), mask)[1])
        e_o = self.relu_point(self.conv2(xyz, xyz, out1, knn_xyz, dist.sqrt(), mask)[1])
        e_o = ME.SparseTensor(features=e_o.squeeze(0), tensor_stride=stride, coordinate_map_key=merged_f.coordinate_map_key,
                              coordinate_manager=merged_f.coordinate_manager)

        ref = self.conv_ref(f2)
        e_c = self.down1(e_o)
        u1 = self.up2(e_c, e_o)
        delta_e = e_o - u1
        e_f = self.down2(delta_e, ref)
        e_c = self.prune(e_c, ref)
        e = e_c + e_f

        # motion compression
        compressed_motion2 = self.motion_compressor(e)
        return compressed_motion2

    def get_motion_vector_2(self, f2, quant_compressed_motion):
        print('get_motion_vector_2 enable')
        ref = self.conv_ref(f2)
        reconstructed_motion1 = self.motion_decompressor1(quant_compressed_motion, ref)

        # motion compensation
        # get motion flow m
        reconstructed_motion2 = self.motion_decompressor2(reconstructed_motion1, f2)
        m_f = self.high_resolution_motion_generator(reconstructed_motion2)

        m_c = self.low_resolution_motion_generator(reconstructed_motion1)
        m_c = self.unpooling(m_c)
        m_c = self.prune(m_c, f2)
        m = m_c + m_f
        return m

    def get_motion_vector(self, f1, f2, stride=8, K=16, r=3):
        # motion estimation
        merged_f = merge_two_frames(f1, f2)
        # out1 = self.relu(self.conv1(merged_f))
        # e_o = self.relu(self.conv2(out1))

        xyz, feature = merged_f.C, merged_f.F
        xyz, feature = (xyz.to(torch.float32)/stride).unsqueeze(0)[:, :, 1:], feature.unsqueeze(0)  # 球KNN去除球面外的邻居，坐标需要先除以步长
        dist, knn_xyz, mask = ball_knn(xyz, xyz, K=16, radius=3)
        xyz = xyz/5  # mlp特征提取中涉及坐标和距离的concate，需要减小坐标以避免坐标值与距离数量级相差太大
        out1 = self.relu_point(self.conv1(xyz, xyz, feature, knn_xyz, dist.sqrt(), mask)[1])
        e_o = self.relu_point(self.conv2(xyz, xyz, out1, knn_xyz, dist.sqrt(), mask)[1])
        e_o = ME.SparseTensor(features=e_o.squeeze(0), tensor_stride=stride, coordinate_map_key=merged_f.coordinate_map_key,
                              coordinate_manager=merged_f.coordinate_manager)

        ref = self.conv_ref(f2)
        e_c = self.down1(e_o)
        u1 = self.up2(e_c, e_o)
        delta_e = e_o - u1
        e_f = self.down2(delta_e, ref)
        e_c = self.prune(e_c, ref)
        e = e_c + e_f

        # motion compression
        compressed_motion2 = self.motion_compressor(e)
        quant_compressed_motion = ME.SparseTensor(quant(compressed_motion2.F, training=self.training),
                                                  coordinate_map_key=compressed_motion2.coordinate_map_key,
                                                  coordinate_manager=compressed_motion2.coordinate_manager)
        reconstructed_motion1 = self.motion_decompressor1(quant_compressed_motion, ref)

        # motion compensation
        # get motion flow m
        reconstructed_motion2 = self.motion_decompressor2(reconstructed_motion1, f2)
        m_f = self.high_resolution_motion_generator(reconstructed_motion2)

        m_c = self.low_resolution_motion_generator(reconstructed_motion1)
        m_c = self.unpooling(m_c)
        m_c = self.prune(m_c, f2)
        m = m_c + m_f
        return quant_compressed_motion, m, compressed_motion2

    def decoder_side(self, quant_motion, f2_coor, ys2_4_coor, s1, s2, coarse=True):
        ys2_4_coor_ = ME.SparseTensor(torch.ones([ys2_4_coor.size(0), 1], dtype=torch.float32, device=ys2_4_coor.device),
                                      ys2_4_coor, tensor_stride=s1)
        f2_coor_ = ME.SparseTensor(torch.ones([f2_coor.size(0), 1], dtype=torch.float32, device=f2_coor.device),
                                   f2_coor, tensor_stride=s2)
        reconstructed_motion1 = self.motion_decompressor1(quant_motion, ys2_4_coor_)

        # motion compensation
        # get motion flow m
        reconstructed_motion2 = self.motion_decompressor2(reconstructed_motion1, f2_coor_)
        m_f = self.high_resolution_motion_generator(reconstructed_motion2)

        m_c = self.low_resolution_motion_generator(reconstructed_motion1)
        m_c = self.unpooling(m_c)
        m_c = self.prune(m_c, f2_coor_)
        m = m_c + m_f
        return m

    def decoder_predict(self, f1, f2_C, m, stride):
        f1 = sort_by_coor_sum(f1)
        xyz1, xyz2, point1 = f1.C / stride, f2_C / stride, f1.F
        xyz1, xyz2, point1 = xyz1[:, 1:], xyz2[:, 1:], point1

        B = 1
        N, _ = f2_C.size()
        C = f1.size()[-1]
        m = sort_by_coor_sum(m, stride)
        motion = m.F
        motion = motion.reshape(N, C, 3)
        xyz2_ = (xyz2.unsqueeze(1) + motion).reshape(-1, 3).unsqueeze(0)  # point_num: 1*N*C
        xyz1, point1 = xyz1.unsqueeze(0), point1.unsqueeze(0)

        # 3DAWI
        # t = time.time()
        dist, knn_index1_, __ = knn_points(xyz2_, xyz1, K=3)
        # print('xyz2_, xyz1', xyz2_.size(), xyz1.size())
        # print('KNN-3', time.time()-t)
        dist += 1e-8
        knn_index1_ = knn_index1_.reshape(B, N, C, 3)
        # t = time.time()
        knn_point1_ = index_by_channel(point1, knn_index1_, 3)
        # print('index by channel', time.time()-t)
        dist = dist.reshape(B, N, C, 3)
        weights = 1 / dist
        weights = weights / torch.clamp(weights.sum(dim=3, keepdim=True), min=3)
        predicted_point2 = (weights * knn_point1_).sum(dim=3).squeeze(0)
        predicted_f2 = ME.SparseTensor(predicted_point2, coordinates=f2_C, coordinate_manager=f1.coordinate_manager,
                                       tensor_stride=stride, device=f1.device)
        return predicted_f2

    def forward(self, f1, f2, m, stride=8):
        # 3DAWI
        f1 = sort_by_coor_sum(f1)
        f2, m = sort_by_coor_sum(f2, stride), sort_by_coor_sum(m, stride)
        motion = m.F
        xyz1, xyz2, point1, point2 = f1.C / stride, m.C / stride, f1.F, f2.F  # m.C == f2.C
        xyz1, xyz2, point1, point2 = xyz1[:, 1:].unsqueeze(0), xyz2[:, 1:].unsqueeze(0), point1.unsqueeze(
            0), point2.unsqueeze(0)
        B, N, C = point2.size()
        motion = motion.reshape(B, N, C, 3)
        xyz2_ = (xyz2.unsqueeze(2) + motion).reshape(B, -1, 3)
        dist, knn_index1_, __ = knn_points(xyz2_, xyz1, K=3)
        dist += 1e-8
        knn_index1_ = knn_index1_.reshape(B, N, C, 3)
        knn_point1_ = index_by_channel(point1, knn_index1_, 3)
        dist = dist.reshape(B, N, C, 3)
        weights = 1 / dist
        weights = weights / torch.clamp(weights.sum(dim=3, keepdim=True), min=3)
        predicted_point2 = (weights * knn_point1_).sum(dim=3).squeeze(0)
        predicted_f2 = ME.SparseTensor(predicted_point2, coordinates=f2.C, coordinate_manager=f2.coordinate_manager,
                                       tensor_stride=stride, device=f2.device)

        # get residual
        residual_f2 = f2 - predicted_f2
        residual_f2 = ME.SparseTensor(residual_f2.F, coordinates=residual_f2.C, tensor_stride=stride, device=f2.device)
        return residual_f2, predicted_f2


class EncoderLayer(nn.Module):
    def __init__(self, input, hidden, output, block_layers, kernel=2, resnet=InceptionResNet):
        super(EncoderLayer, self).__init__()
        self.resnet = resnet
        self.conv = ME.MinkowskiConvolution(
            in_channels=input,
            out_channels=hidden,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.down = ME.MinkowskiConvolution(
            in_channels=hidden,
            out_channels=output,
            kernel_size=kernel,
            stride=2,
            bias=True,
            dimension=3)
        if resnet is not None:
            self.block = self.make_layer(resnet, block_layers, output)
        self.relu = ME.MinkowskiReLU()

    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.down(self.relu(self.conv(x)))
        if self.resnet is not None:
            out = self.block(self.relu(out))
        return out


class DownsampleWithPruning(nn.Module):
    def __init__(self, input, output, block_layers, kernel=2, resnet=InceptionResNet):
        super(DownsampleWithPruning, self).__init__()
        self.resnet = resnet
        self.down = ME.MinkowskiConvolution(
            in_channels=input,
            out_channels=output,
            kernel_size=kernel,
            stride=2,
            bias=True,
            dimension=3)
        if resnet is not None:
            self.block = self.make_layer(resnet, block_layers, output)
        self.relu = ME.MinkowskiReLU()
        self.pruning = ME.MinkowskiPruning()

    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))

        return nn.Sequential(*layers)

    def forward(self, x, ref=None):
        out = self.down(x)
        if self.resnet is not None:
            out = self.block(self.relu(out))
        if ref is not None:
            mask = get_target_by_sp_tensor(out, ref)
            out = self.pruning(out, mask.to(out.device))
        return out


class DecoderLayer(nn.Module):
    def __init__(self, input, hidden, output, block_layers, kernel=2):
        super(DecoderLayer, self).__init__()
        self.up = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=input,
            out_channels=hidden,
            kernel_size=kernel,
            stride=2,
            bias=True,
            dimension=3)
        self.conv = ME.MinkowskiConvolution(
            in_channels=hidden,
            out_channels=output,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv_res = ME.MinkowskiConvolution(
            in_channels=input,
            out_channels=output,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        self.block = self.make_layer(
            InceptionResNet, block_layers, output)
        self.conv_cls = ME.MinkowskiConvolution(
            in_channels=output,
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.pruning = ME.MinkowskiPruning()
        self.relu = ME.MinkowskiReLU()

    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))
        return nn.Sequential(*layers)

    def get_cls(self, x):
        out = self.relu(self.conv(self.relu(self.up(x))))
        out = self.block(out)
        out_cls = self.conv_cls(out)
        return out_cls

    def prune(self, f1, f2):
        mask = get_target_by_sp_tensor(f1, f2)
        out = self.pruning(f1, mask)
        return out

    def deconv_with_prune(self, x, target_label):
        out = self.relu(self.conv(self.relu(self.up(x))))
        out = self.block(out)
        out_pruned = self.prune(out, target_label)
        return out_pruned

    def evaluate(self, x, adaptive, num_points, rho=1, residual=None, lossless=False):
        training = self.training
        out = self.relu(self.conv(self.relu(self.up(x))))
        out = self.block(out)
        # if residual is not None:
        #     stride = out.tensor_stride[0]
        #     residual = ME.SparseTensor(residual.F, coordinates=residual.C,
        #                                coordinate_manager=out.coordinate_manager)
        #     out = out + residual
        #     out = ME.SparseTensor(out.F, coordinates=out.C, tensor_stride=stride)
        out_cls = self.conv_cls(out)

        if adaptive:
            coords_nums = num_points
            keep = keep_adaptive(out_cls, coords_nums, rho=rho)
        else:
            keep = (out_cls.F > 0).squeeze()
            if out_cls.F.max() < 0:
                # keep at least one points.
                print('===0; max value < 0', out_cls.F.max())
                _, idx = torch.topk(out_cls.F.squeeze(), 1)
                keep[idx] = True

        # If training, force target shape generation, use net.eval() to disable

        # Remove voxels
        out_pruned = self.pruning(out, keep.to(out.device))
        return out_pruned, out_cls, keep

    def forward(self, x, target_label, adaptive, rho=1, residual=None, lossless=False):
        training = self.training
        out = self.relu(self.conv(self.relu(self.up(x))))
        out = self.block(out)
        if residual is not None:  # only for dec1, add prediction
            residual = ME.SparseTensor(residual.F, coordinates=residual.C,
                                       coordinate_manager=out.coordinate_manager)
            out = out + residual
            out = ME.SparseTensor(out.F, coordinates=out.C, tensor_stride=x.tensor_stride[0]//2)
        out_cls = self.conv_cls(out)
        target = get_target_by_sp_tensor(out, target_label)

        if adaptive:
            coords_nums = [len(coords) for coords in target_label.decomposed_coordinates]
            keep = keep_adaptive(out_cls, coords_nums, rho=rho)
        else:
            keep = (out_cls.F > 0).squeeze()
            if out_cls.F.max() < 0:
                # keep at least one points.
                print('===0; max value < 0', out_cls.F.max())
                _, idx = torch.topk(out_cls.F.squeeze(), 1)
                keep[idx] = True

        # If training, force target shape generation, use net.eval() to disable
        if training or residual is not None:
            keep += target
        elif lossless:
            keep = target

        # Remove voxels
        out_pruned = self.pruning(out, keep.to(out.device))
        return out_pruned, out_cls, target, keep


class DeconvWithPruning(nn.Module):
    def __init__(self, input, output):
        super(DeconvWithPruning, self).__init__()
        self.up = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=input,
            out_channels=output,
            kernel_size=3,
            stride=2,
            bias=True,
            dimension=3)
        self.pruning = ME.MinkowskiPruning()
        self.relu = ME.MinkowskiReLU()

    def forward(self, x, ref=None):
        out = self.up(x)
        if ref is not None:
            mask = get_target_by_sp_tensor(out, ref)
            out = self.pruning(out, mask.to(x.device))
        return out


class Bitparm(nn.Module):
    # save params
    def __init__(self, channel, dimension=4, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        para = [1 for i in range(dimension)]
        para[dimension - 1] = -1
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(para), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(para), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(para), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)


class BitEstimator(nn.Module):
    def __init__(self, channel, dimension=3):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel, dimension=dimension)
        self.f2 = Bitparm(channel, dimension=dimension)
        self.f3 = Bitparm(channel, dimension=dimension)
        self.f4 = Bitparm(channel, dimension=dimension, final=True)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)


# only for losslessly coordinate coding. e.g. f2 coord
class LosslessCompressor(nn.Module):
    def __init__(self):
        super(LosslessCompressor, self).__init__()
        self.compressor1 = EncoderLayer(1, 16, 32, 3)
        # self.compressor2 = EncoderLayer(16, 16, 16, 3)
        self.compressor2 = ME.MinkowskiConvolution(in_channels=32, out_channels=4, kernel_size=3,
                                                   stride=1,
                                                   bias=True,
                                                   dimension=3)
        # self.decompressor1 = DecoderLayer1(4, 16, 16, 3, kernel=2, resnet=InceptionResNet)
        self.decompressor1 = DecoderLayer(4, 16, 32, 3, kernel=2)
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        self.relu = ME.MinkowskiLeakyReLU(inplace=True)
        self.bitEstimator = BitEstimator(4, 3)

    def get_cls(self, pc):
        return self.decompressor1.get_cls(pc)

    def forward(self, pc, num_points, sort_coordinates=False):
        out1 = self.compressor1(pc)
        out2 = self.compressor2(out1)
        quant_out2 = ME.SparseTensor(quant(out2.F, training=self.training),
                                     coordinate_map_key=out2.coordinate_map_key,
                                     coordinate_manager=out2.coordinate_manager,
                                     device=out2.device)
        if sort_coordinates:
            quant_out2 = sort_by_coor_sum(quant_out2, 8)
        out3, cls, target, keep = self.decompressor1(quant_out2, pc, True)
        bits1 = self.bce(cls.F.squeeze(),
                         target.type(cls.F.dtype).to(pc.device)) / math.log(2)
        p = self.bitEstimator(quant_out2.F + 0.5) - self.bitEstimator(quant_out2.F - 0.5)
        bits = torch.sum(torch.clamp(-1.0 * torch.log(p + 1e-10) / math.log(2.0), 0, 50))
        bits = bits1 + bits
        return bits, quant_out2, cls, target
