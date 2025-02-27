import collections
import open3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from models.model_utils import *
# from models.model_utils_for_test import *
# from dataset_lossy import *

class get_model(nn.Module):
    def __init__(self, channels=8):
        super(get_model, self).__init__()
        self.enc1 = EncoderLayer(1, 16, 32, 3)
        self.enc2 = EncoderLayer(32, 32, 64, 3)
        self.inter_prediction_c = inter_prediction(64, 64, 48)
        self.inter_prediction_f = inter_prediction(64, 64, 48)
        self.enc3 = EncoderLayer(64, 64, 32, 3)
        self.enc4 = ME.MinkowskiConvolution(in_channels=32, out_channels=channels, kernel_size=3, stride=1, bias=True, dimension=3)

        self.dec1 = DecoderLayer(channels, 64, 64, 3)  # (64, 64, 64, 3)
        self.dec2 = DecoderLayer(64, 32, 32, 3)
        self.dec3 = DecoderLayer(32, 16, 16, 3)

        self.BitEstimator = BitEstimator(channels, 3)
        self.MotionBitEstimator_c = BitEstimator(48, 3)
        self.MotionBitEstimator_f = BitEstimator(48, 3)
        self.crit = torch.nn.BCEWithLogitsLoss()

        self.enc0 = EncoderLayer(64, 64, 64, 3)
        self.dec0 = DecoderLayer(64, 64, 64, 3)
        self.enc_extra = EncoderLayer(64, 64, 64, 3)
        self.up_extra = DeconvWithPruning(64 * 3, 64 * 3)

        self.dec_r = DecoderLayer(channels, 32, 64, 3)
        self.compress_r = ME.MinkowskiConvolution(in_channels=64, out_channels=channels, kernel_size=1, stride=1,
                                                  bias=True, dimension=3)
        self.decompress_r = ME.MinkowskiConvolution(in_channels=channels, out_channels=64, kernel_size=1, stride=1,
                                                  bias=True, dimension=3)

    def forward(self, f1, f2, device, epoch=99999, show_motion=False, rho=1):
        num_points = f2.C.size(0)
        ys1, ys2 = [f1, 0, 0, 0, 0, 0], [f2, 0, 0, 0, 0, 0]
        out2, out_cls2, target2, keep2 = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]

        # feature extraction
        ys1[1] = self.enc1(ys1[0])
        ys1[2] = self.enc2(ys1[1])
        ys2[1] = self.enc1(ys2[0])
        ys2[2] = self.enc2(ys2[1])
        ys1[3] = self.enc0(ys1[2])  # down-sampling 3
        ys2[3] = self.enc0(ys2[2])

        # inter prediction
        # coarse, 3x down
        quant_motion_1, m_1, compressed_m_1 = self.inter_prediction_c.get_motion_vector(ys1[3], ys2[3], stride=8)
        m_1 = self.up_extra(m_1, ys2[2])
        _, predicted_point_c = self.inter_prediction_c(ys1[2], ys2[2], m_1, stride=4)
        # fine, 2x down
        quant_motion_2, m_2, compressed_m_2 = self.inter_prediction_f.get_motion_vector(predicted_point_c, ys2[2], stride=4)
        residual, predicted_point2 = self.inter_prediction_f(predicted_point_c, ys2[2], m_2, stride=4)
        # print('motion:', torch.round(compressed_m_1.F).abs().sum().item(), torch.round(compressed_m_2.F).abs().sum().item())

        # residual compression
        ys2[4] = self.enc3(residual)
        ys2[5] = self.enc4(ys2[4])
        '''3x down residual'''
        quant_y = quant(ys2[5].F.unsqueeze(0), training=self.training)
        # p = self.BitEstimator(ys2[5].F.unsqueeze(0)+0.5) - self.BitEstimator(ys2[5].F.unsqueeze(0)-0.5)
        p = self.BitEstimator(quant_y + 0.5) - self.BitEstimator(quant_y - 0.5)
        bits = torch.sum(torch.clamp(-1.0 * torch.log(p + 1e-10) / math.log(2.0), 0, 50))
        '''motion'''
        quant_motion_F = [quant_motion_1.F.unsqueeze(0), quant_motion_2.F.unsqueeze(0)]
        motion_p_1 = self.MotionBitEstimator_c(quant_motion_F[0] + 0.5) - self.MotionBitEstimator_c(quant_motion_F[0] - 0.5)
        motion_bits_1 = torch.sum(torch.clamp(-1.0 * torch.log(motion_p_1 + 1e-10) / math.log(2.0), 0, 50))
        motion_p_2 = self.MotionBitEstimator_f(quant_motion_F[1] + 0.5) - self.MotionBitEstimator_f(quant_motion_F[1] - 0.5)
        motion_bits_2 = torch.sum(torch.clamp(-1.0 * torch.log(motion_p_2 + 1e-10) / math.log(2.0), 0, 50))
        motion_bits = motion_bits_1 + motion_bits_2
        factor = 0.95
        if self.training:
            motion_bits = factor * motion_bits
        bpp = (bits + motion_bits) / num_points

        # point cloud reconstruction
        y2_recon = ME.SparseTensor(quant_y.squeeze(0), coordinate_map_key=ys2[5].coordinate_map_key,
                                   coordinate_manager=ys2[5].coordinate_manager, device=ys2[5].device)
        out2[0], out_cls2[0], target2[0], keep2[0] = self.dec1(y2_recon, ys2[2], True, residual=predicted_point2)
        out2[1], out_cls2[1], target2[1], keep2[1] = self.dec2(out2[0], ys2[1], True, 1 if self.training else 1)
        out2[2], out_cls2[2], target2[2], keep2[2] = self.dec3(out2[1], ys2[0], True, rho)

        if show_motion:
            print('residual:', residual.F.abs().sum().item())
            print('motion:', torch.round(compressed_m_1.F).abs().sum().item(),
                  torch.round(compressed_m_2.F).abs().sum().item())
            print('motion bpp', motion_bits_1/num_points, motion_bits_2/num_points)
            print('residual bpp', bits/num_points)

            # m_2 = ME.SparseTensor(m_2.F, coordinates=m_2.C, coordinate_manager=m_1.coordinate_manager)
            # m = m_1 + m_2
            # xyz = m.C[:, 1:]
            # color = m.F.reshape(-1, 64, 3)
            # color = color[:, 35]
            # c_max, c_min = 7, -7
            # color = (color - c_min) / (c_max - c_min)
            # # color = (color - color.min(dim=0, keepdim=True)[0]) / (
            # #         color.max(dim=0, keepdim=True)[0] - color.min(dim=0, keepdim=True)[0])
            # recon_pcd = open3d.geometry.PointCloud()
            # recon_pcd.points = open3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
            # recon_pcd.colors = open3d.utility.Vector3dVector(color.detach().cpu().numpy())
            # open3d.io.write_point_cloud('motion.ply', recon_pcd, write_ascii=True)
            #
            # xyz = m_1.C[:, 1:]
            # color = m_1.F.reshape(-1, 64, 3)
            # color = color[:, 35]
            # # print(color.max(dim=0, keepdim=True)[0], color.min(dim=0, keepdim=True)[0])
            # c_max, c_min = 7, -7
            # color = (color - c_min) / (c_max - c_min)
            # # color = (color - color.min(dim=0, keepdim=True)[0]) / (
            # #         color.max(dim=0, keepdim=True)[0] - color.min(dim=0, keepdim=True)[0])
            # recon_pcd = open3d.geometry.PointCloud()
            # recon_pcd.points = open3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
            # recon_pcd.colors = open3d.utility.Vector3dVector(color.detach().cpu().numpy())
            # open3d.io.write_point_cloud('motion-1.ply', recon_pcd, write_ascii=True)
            #
            # xyz = m_2.C[:, 1:]
            # color = m_2.F.reshape(-1, 64, 3)
            # color = color[:, 35]
            # # color = torch.mean(color, 1)
            # # print('color', color.max(dim=0, keepdim=True)[0], color.min(dim=0, keepdim=True)[0])
            # color = (color - color.min(dim=0, keepdim=True)[0]) / (
            #         color.max(dim=0, keepdim=True)[0] - color.min(dim=0, keepdim=True)[0])
            # recon_pcd = open3d.geometry.PointCloud()
            # recon_pcd.points = open3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
            # recon_pcd.colors = open3d.utility.Vector3dVector(color.detach().cpu().numpy())
            # open3d.io.write_point_cloud('motion-2.ply', recon_pcd, write_ascii=True)

        return ys2, out2, out_cls2, target2, keep2, bpp

