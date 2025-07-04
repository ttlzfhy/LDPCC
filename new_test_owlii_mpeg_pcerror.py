"""
    Compared with new_test_owlii_mpeg.py
    use mpeg_dmetric to calculate PSNR rather than manual function.
"""
import argparse
import importlib
import logging
import sys
import os
from pathlib import Path


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Test Script')
    parser.add_argument('--model', type=str, default='DDPCC_geo')
    parser.add_argument('--lossless_model', type=str, default='DDPCC_lossless_coder')
    parser.add_argument('--log_name', type=str, default='bbb')
    parser.add_argument('--gpu', type=str, default='2', help='specify gpu device [default: 0]')
    parser.add_argument('--channels', default=8, type=int)
    parser.add_argument('--ckpt_dir', type=str,
                        default='./final_ckpts')
    parser.add_argument('--pcgcv2_ckpt_dir', type=str,
                        default='./pcgcv2_ckpts')
    parser.add_argument('--frame_count', type=int, default=100, help='number of frames to be coded')

    parser.add_argument('--overwrite', type=bool, default=False, help='overwrite the bitstream of previous frame')
    parser.add_argument('--dataset_dir', type=str, default='/home/data/dataset/point_cloud/aipcc_cfp_testdata_all/model_vox10')
    parser.add_argument('--f1_path', default=None, type=str, help='path of the first frame, when test was unexpectedly stopped and need continue')
    return parser.parse_args()
args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, './PCGCv2'))

from models.model_utils import index_points, sort_by_coor_sum, coordinate_sort_by_coor_sum, quant
from dataset_lossy_v1 import *
from models.entropy_coding import *
from GPCC.gpcc_wrapper import *
from PCGCv2.eval import test_one_frame
import pandas as pd
import collections, math
from pytorch3d.ops import knn_points


def write_file(x_ori, x_dec, showdir, idx):
    # write files
    if isinstance(x_ori, ME.SparseTensor):
        points_ori = x_ori.C[:, 1:].detach().cpu().numpy()
    else:
        points_ori = x_ori
    # ori_dir = os.path.join(showdir, 'f2'+str(idx)+'.ply')
    ori_dir = os.path.join(showdir, 'f2.ply')
    write_ply_open3d_normal(ori_dir, points_ori)
    #
    if isinstance(x_dec, ME.SparseTensor):
        points_dec = x_dec.C[:, 1:].detach().cpu().numpy()
    else:
        points_dec = x_dec
    dec_dir = os.path.join(showdir, 'f2_dec'+str(idx)+'.ply')
    write_ply_data(dec_dir, points_dec)

    return ori_dir, dec_dir


def psnr_peak_value(x: torch.Tensor):
    if isinstance(x, torch.Tensor):
        coordinates = x.detach().cpu().numpy()
    elif isinstance(x, ME.SparseTensor):
        coordinates = x.C[:, 1:].detach().cpu().numpy()
    else:
        raise TypeError("Input must be torch.Tensor or ME.SparseTensor")
    resolution = 2 ** np.round(np.log2(coordinates.max() - coordinates.min()))
    return int(resolution)


def log_string(string):
    logger.info(string)
    print(string)


def PSNR(pc1, pc2, n1, return_all=False, peak_value=1023):
    pc1, pc2 = pc1.to(torch.float32), pc2.to(torch.float32)
    dist1, knn1, _ = knn_points(pc1, pc2, K=4)  # neighbors of pc1 from pc2
    dist2, knn2, _ = knn_points(pc2, pc1, K=4)  # neighbors of pc2 from pc1
    mask1 = (dist1 == dist1[:, :, :1])
    mask2 = (dist2 == dist2[:, :, :1])
    dist = max(dist1[:, :, 0].mean(), dist2[:, :, 0].mean())  # dists from knn_points are squared dists
    cd = max(dist1[:, :, 0].sqrt().mean(), dist2[:, :, 0].sqrt().mean())
    # print(pc1, pc2)
    d1_psnr = 10*math.log(3*peak_value*peak_value/dist)/math.log(10)
    d1_psnr_1 = 10 * math.log(3 * peak_value * peak_value / dist1[:, :, 0].mean()) / math.log(10)
    d1_psnr_2 = 10 * math.log(3 * peak_value * peak_value / dist2[:, :, 0].mean()) / math.log(10)
    knn1_ = knn1.reshape(-1)
    n1_src = (n1.unsqueeze(2).repeat(1, 1, 4, 1)*(mask1.unsqueeze(-1))).reshape(-1, 3)
    n2 = torch.zeros_like(pc2.squeeze(0), dtype=torch.float64)
    n2.index_add_(0, knn1_, n1_src)
    n2 = n2.reshape(1, -1, 3)

    n2_counter = torch.zeros(pc2.size()[1], dtype=torch.float32, device=pc2.device)
    counter_knn1 = knn1.reshape(-1)
    n1_counter_src = mask1.reshape(-1).to(torch.float32)
    n2_counter.index_add_(0, counter_knn1, n1_counter_src)
    n2_counter = n2_counter.reshape(1, -1, 1)
    n2_counter += 0.00000001

    n2 /= n2_counter

    v2 = index_points(pc1, knn2) - pc2.unsqueeze(2)
    n2_ = index_points(n1, knn2)
    n21 = (n2_*(mask2.unsqueeze(-1))).sum(dim=2) / (mask2.sum(dim=-1, keepdim=True))
    n2 += (n2_counter < 0.0001) * n21

    d2_ = (((v2*n2_).sum(dim=-1).square()*mask2).sum(dim=-1)/mask2.sum(dim=-1)).mean()
    v1 = index_points(pc2, knn1) - pc1.unsqueeze(2)
    n1_ = index_points(n2, knn1)
    d1_ = (((v1 * n1_).sum(dim=-1).square() * mask1).sum(dim=-1) / mask1.sum(dim=-1)).mean()
    dist_ = max(d1_, d2_)
    d2_psnr = 10*math.log(3*peak_value*peak_value/dist_)/math.log(10)
    # print(d1_psnr, d2_psnr)
    if return_all:
        return d1_psnr_1, d1_psnr_2, d1_psnr, d2_psnr, cd.item()
    else:
        return d1_psnr, d2_psnr, cd.item()


def encode(f1, f2, bitstream_filename, gpcc_bitstream_filename):
    ys1, ys2 = [f1, 0, 0, 0, 0, 0], [f2, 0, 0, 0, 0, 0]

    # feature extraction
    ys1[1] = model.enc1(ys1[0])
    ys1[2] = model.enc2(ys1[1])
    ys2[1] = model.enc1(ys2[0])
    ys2[2] = model.enc2(ys2[1])
    ys1[3] = model.enc0(ys1[2])  # down-sampling 3
    ys2[3] = model.enc0(ys2[2])

    # inter prediction
    # coarse, 3×down
    quant_motion_1, m_1, compressed_m_1 = model.inter_prediction_c.get_motion_vector(ys1[3], ys2[3], stride=8)
    m_1 = model.up_extra(m_1, ys2[2])
    _, predicted_point_c = model.inter_prediction_c(ys1[2], ys2[2], m_1, stride=4)
    # fine, 2×down
    quant_motion_2, m_2, compressed_m_2 = model.inter_prediction_f.get_motion_vector(predicted_point_c, ys2[2], stride=4)
    residual, predicted_point2 = model.inter_prediction_f(predicted_point_c, ys2[2], m_2, stride=4)

    quant_motion_1, quant_motion_2 = sort_by_coor_sum(quant_motion_1, 32), sort_by_coor_sum(quant_motion_2, 16)
    '''
    print('enc-m:', m_1.F.sum().item(), m_2.F.sum().item())
    print('enc-predict', predicted_point_c.F.sum().item(), predicted_point2.F.sum().item())'''

    # residual compression
    ys2[4] = model.enc3(residual)
    ys2[5] = model.enc4(ys2[4])
    ys2[5] = sort_by_coor_sum(ys2[5], 8)
    quant_y = quant(ys2[5].F.unsqueeze(0), training=model.training)

    # encode C_{x_t}^2
    ys2_2 = ME.SparseTensor(torch.ones_like(ys2[2].C[:, :1], dtype=torch.float32), coordinates=ys2[2].C,
                            tensor_stride=4)
    ys2_2 = sort_by_coor_sum(ys2_2, 4)
    _, ys2_2_feature, ys2_2_cls, ys2_2_target = lossless_model.compressor(ys2_2, ys2_2.size()[0],
                                                                          sort_coordinates=True)
    p = torch.sigmoid(ys2_2_cls.F)
    # print('enc-ys2_2', ys2_2.C.sum().item(), ys2_2_cls.F.sum().item(), ys2_2_feature.F.sum().item())

    # entropy coding
    motion_bitstream_1, min_v_motion_1, max_v_motion_1 = factorized_entropy_coding(model.MotionBitEstimator_c,
                                                                                   quant_motion_1.F.unsqueeze(0))
    motion_bitstream_2, min_v_motion_2, max_v_motion_2 = factorized_entropy_coding(model.MotionBitEstimator_f,
                                                                                   quant_motion_2.F.unsqueeze(0))
    residual_bitstream, min_v_res, max_v_res = factorized_entropy_coding(model.BitEstimator, quant_y)
    ys2_2_feature_bitstream, min_v_res2, max_v_res2 = factorized_entropy_coding(
        lossless_model.compressor.bitEstimator, ys2_2_feature.F)
    ys2_2_bitstream = binary_entropy_coding(p, ys2_2_target)
    ys2_5_C = ys2[5].decomposed_coordinates[0].detach().cpu().numpy()
    write_ply_data(os.path.join(showdir, 'ys2_5.ply'), ys2_5_C / 8)
    gpcc_encode(os.path.join(showdir, 'ys2_5.ply'), gpcc_bitstream_filename)
    file = open(bitstream_filename, 'wb')
    file.write(np.array(min_v_motion_1, dtype=np.int16).tobytes())
    file.write(np.array(max_v_motion_1, dtype=np.int16).tobytes())
    file.write(np.array(min_v_motion_2, dtype=np.int16).tobytes())
    file.write(np.array(max_v_motion_2, dtype=np.int16).tobytes())
    file.write(np.array(min_v_res, dtype=np.int16).tobytes())
    file.write(np.array(max_v_res, dtype=np.int16).tobytes())
    file.write(np.array(min_v_res2, dtype=np.int16).tobytes())
    file.write(np.array(max_v_res2, dtype=np.int16).tobytes())
    file.write(np.array(quant_y.shape[1], dtype=np.uint32).tobytes())
    file.write(np.array(quant_motion_1.shape[0], dtype=np.uint32).tobytes())
    file.write(np.array(quant_motion_2.shape[0], dtype=np.uint32).tobytes())
    file.write(np.array(ys2[0].shape[0], dtype=np.uint32).tobytes())
    file.write(np.array(ys2[1].shape[0], dtype=np.uint32).tobytes())
    file.write(np.array(len(motion_bitstream_1), dtype=np.uint32).tobytes())
    file.write(np.array(len(motion_bitstream_2), dtype=np.uint32).tobytes())
    file.write(np.array(len(ys2_2_feature_bitstream), dtype=np.uint32).tobytes())
    file.write(np.array(len(ys2_2_bitstream), dtype=np.uint32).tobytes())
    file.write(motion_bitstream_1)
    file.write(motion_bitstream_2)
    file.write(ys2_2_feature_bitstream)
    file.write(ys2_2_bitstream)
    file.write(residual_bitstream)
    file.close()
    
    m_2 = ME.SparseTensor(m_2.F, coordinates=m_2.C, coordinate_manager=m_1.coordinate_manager)
    m = m_1 + m_2
    
    return {'motion_bits': len(motion_bitstream_1) * 8 + len(motion_bitstream_2) * 8,
            'residual_bits': len(residual_bitstream) * 8,
            'ys2_2_lossless_bits': len(ys2_2_feature_bitstream) * 8 + len(ys2_2_bitstream) * 8,
            'round_motion': quant_motion_1.F.abs().sum().item() + quant_motion_2.F.abs().sum().item(), 
            'round_residual': quant_y.abs().sum().item(), 
            'motion_vector': m}


def decode(f1, bitstream_filename, gpcc_bitstream_filename):
    ys1 = [f1, 0, 0, 0]
    file = open(bitstream_filename, 'rb')
    min_v_motion_1_, max_v_motion_1_, min_v_motion_2_, max_v_motion_2_, min_v_res_, max_v_res_, min_v_res2_, max_v_res2_ \
        = np.frombuffer(file.read(16), dtype=np.int16)
    quant_y_length, quant_motion_1_length, quant_motion_2_length = np.frombuffer(
        file.read(12), dtype=np.uint32)
    num_points_0, num_points_1 = np.frombuffer(
        file.read(8), dtype=np.uint32)
    motion_bitstream_1_length, motion_bitstream_2_length, ys2_2_feature_bitstream_length, ys2_2_bitstream_length = np.frombuffer(
        file.read(16), dtype=np.uint32)
    motion_bitstream_1_ = file.read(motion_bitstream_1_length)
    motion_bitstream_2_ = file.read(motion_bitstream_2_length)
    ys2_2_feature_bitstream_ = file.read(ys2_2_feature_bitstream_length)
    ys2_2_bitstream_ = file.read(ys2_2_bitstream_length)
    residual_bitstream_ = file.read()
    ys1[1] = model.enc1(ys1[0])
    ys1[2] = model.enc2(ys1[1])
    ys1[3] = model.enc0(ys1[2])
    quant_y_F = factorized_entropy_decoding(model.BitEstimator, [quant_y_length, 8],
                                            residual_bitstream_,
                                            min_v_res_, max_v_res_, device).to(device).to(torch.float32)
    quant_motion_F_1_ = factorized_entropy_decoding(model.MotionBitEstimator_c, [quant_motion_1_length, 48],
                                                  motion_bitstream_1_, min_v_motion_1_, max_v_motion_1_,
                                                  device).to(device).to(torch.float32)
    quant_motion_F_2_ = factorized_entropy_decoding(model.MotionBitEstimator_f, [quant_motion_2_length, 48],
                                                    motion_bitstream_2_, min_v_motion_2_, max_v_motion_2_,
                                                    device).to(device).to(torch.float32)

    ys2_2_feature_F = factorized_entropy_decoding(lossless_model.compressor.bitEstimator,
                                                  [quant_y_length, 4], ys2_2_feature_bitstream_,
                                                  min_v_res2_, max_v_res2_, device).to(device).to(
                                                  torch.float32)
    # print('decoder m:', quant_motion_F_1_.sum().item(), quant_motion_F_2_.sum().item(),
    #       quant_y_F.sum().item(), ys2_2_feature_F.sum().item())

    gpcc_decode(gpcc_bitstream_filename, os.path.join(showdir, 'recon_ys2_5.ply'))
    recon_ys2_5_C = 8 * torch.tensor(read_point_cloud(os.path.join(showdir, 'recon_ys2_5.ply')),
                                     dtype=torch.int32, device=device)
    recon_ys2_5_C = torch.cat([torch.zeros_like(recon_ys2_5_C[:, :1]), recon_ys2_5_C], dim=-1)
    recon_ys2_5_C = coordinate_sort_by_coor_sum(recon_ys2_5_C)
    # print('recon_ys2_5_C', recon_ys2_5_C.sum().item())
    recon_ys2_2_feature = ME.SparseTensor(ys2_2_feature_F, coordinates=recon_ys2_5_C, tensor_stride=8)
    recon_ys2_2_cls = lossless_model.compressor.get_cls(recon_ys2_2_feature)
    # print('dec-ys2_2', recon_ys2_2_cls.F.sum().item(), recon_ys2_2_feature.F.sum().item())
    recon_p = torch.sigmoid(recon_ys2_2_cls.F)
    ys2_2_mask = binary_entropy_decoding(recon_p, ys2_2_bitstream_).to(torch.bool).to(device)
    recon_ys2_2_C = ME.MinkowskiPruning()(recon_ys2_2_cls, ys2_2_mask).C  # c2
    # print('dec-ys2_2_c', recon_ys2_2_C.sum().item())
    y2_recon_ = ME.SparseTensor(quant_y_F, coordinates=recon_ys2_5_C, tensor_stride=8)  # residual
    motion_coor_2 = model.inter_prediction_f.get_downsampled_coordinate(recon_ys2_5_C, 8,
                                                                        return_sorted=True)  # 4x down
    motion_coor_1 = model.inter_prediction_c.get_downsampled_coordinate(motion_coor_2, 16,
                                                                        return_sorted=True)  # 5x down
    recon_quant_motion_1 = ME.SparseTensor(quant_motion_F_1_, coordinates=motion_coor_1, tensor_stride=32)
    recon_quant_motion_2 = ME.SparseTensor(quant_motion_F_2_, coordinates=motion_coor_2, tensor_stride=16)
    m_1 = model.inter_prediction_c.decoder_side(recon_quant_motion_1, recon_ys2_5_C, motion_coor_2,
                                                s1=16, s2=8, coarse=True)
    m_2 = model.inter_prediction_f.decoder_side(recon_quant_motion_2, recon_ys2_2_C, recon_ys2_5_C, s1=8, s2=4)
    ys2_2 = ME.SparseTensor(coordinates=recon_ys2_2_C, features=torch.ones_like(recon_ys2_2_C[:, :1], dtype=torch.float32),
                            tensor_stride=4, device=device)
    # print('dec-ys2_2', ys2_2.C.sum().item())
    ys2_2 = sort_by_coor_sum(ys2_2, 4)
    m_1 = model.up_extra(m_1, ys2_2)
    predicted_point_c = model.inter_prediction_c.decoder_predict(ys1[2], ys2_2.C, m_1, stride=4)
    recon_predicted_f2 = model.inter_prediction_f.decoder_predict(predicted_point_c, ys2_2.C, m_2, stride=4)

    # point cloud reconstruction
    out2[0], out_cls2[0], target2[0], keep2[0] = model.dec1(y2_recon_, recon_predicted_f2, True,
                                                            residual=recon_predicted_f2)
    out2[1], out_cls2[1], keep2[1] = model.dec2.evaluate(out2[0], True, [num_points_1], 1)
    out2[2], out_cls2[2], keep2[2] = model.dec3.evaluate(out2[1], True, [num_points_0], 1)

    recon_f2 = ME.SparseTensor(torch.ones_like(out2[2].F[:, :1]), coordinates=out2[2].C)
    recon_f2_C = recon_f2.decomposed_coordinates[0].detach().cpu().numpy()
    f2_C = f2.decomposed_coordinates[0].detach().cpu().numpy()
    # print('----'*10)
    return recon_f2_C, f2_C, recon_f2


def show_motion(motion, showdir, idx):
    # write ply
    xyz = motion.C[:, 1:]
    color = motion.F.reshape(-1, 64, 3)
    color = color[:, 20]
    # print(color.max(dim=0, keepdim=True)[0], color.min(dim=0, keepdim=True)[0])
    # c_max, c_min = 7, -7
    # color = (color - c_min) / (c_max - c_min)
    color = (color - color.min(dim=0, keepdim=True)[0]) / (
            color.max(dim=0, keepdim=True)[0] - color.min(dim=0, keepdim=True)[0])
    recon_pcd = open3d.geometry.PointCloud()
    recon_pcd.points = open3d.utility.Vector3dVector(xyz.detach().cpu().numpy())
    recon_pcd.colors = open3d.utility.Vector3dVector(color.detach().cpu().numpy())
    open3d.io.write_point_cloud(os.path.join(showdir, 'motion' + str(idx) + '.ply'), recon_pcd, write_ascii=True)


if __name__ == '__main__':
    device = torch.device('cuda')
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('./%s.txt' % args.log_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # load model
    log_string('PARAMETER ...')
    log_string(args)
    MODEL = importlib.import_module(args.model)
    model = MODEL.get_model(channels=args.channels)
    model.eval()

    LOSSLESS_MODEL = importlib.import_module(args.lossless_model)
    lossless_model = LOSSLESS_MODEL.get_model()
    lossless_checkpoint = torch.load(os.path.join(args.ckpt_dir, 'lossless_coder.pth'))
    old_paras = lossless_model.state_dict()
    new_state_dict = collections.OrderedDict()
    for k, v in lossless_checkpoint['model_state_dict'].items():
        k1 = k.replace('module.', '')
        if k1 in old_paras:
            new_state_dict[k1] = v
    old_paras.update(new_state_dict)
    lossless_model.load_state_dict(old_paras)
    # lossless_model.load_state_dict(lossless_checkpoint['model_state_dict'])
    lossless_model = lossless_model.to(device).eval()

    ckpts = {
        'new-I2_6.pth': 'r4_0.15bpp.pth',
        'new-I4-5-1.pth': 'r4_0.15bpp.pth',
        'new-I5-15.pth': 'r4_0.15bpp.pth',
        'new-I7-15.pth': 'r5_0.25bpp.pth',
        'new-I9.pth': 'r6_0.3bpp.pth',
        'new-I11.pth': 'r6_0.3bpp.pth',
        'new-I15.pth': 'r7_0.4bpp.pth'
    }
    with torch.no_grad():
        ## dataset settings
        start_frame = 0 if not args.f1_path else int(args.f1_path.split('_')[-1].split('.')[0])
                    
        filedir_list = sorted(glob.glob(os.path.join(args.dataset_dir, '**', '*.*'), recursive=True))
        filedir_list = [f for f in filedir_list if f.endswith('npy') or f.endswith('ply')]
        args.frame_count = min(args.frame_count, len(filedir_list))
        filedir_list = filedir_list[start_frame: args.frame_count]
        
        # save settings
        seqname = os.path.basename(args.dataset_dir)
        test_dirname = 'EncDec_'+str(args.model)+'_'+seqname
        resultdir = os.path.join(BASE_DIR, 'Result_'+test_dirname)
        resultdir_ = Path(resultdir)
        resultdir_.mkdir(exist_ok=True)
        
        all_rate_mean_results = pd.DataFrame([])
        all_mean_csvfile = os.path.join(resultdir, seqname
                            + '_f' + str(start_frame) + '-' + str(args.frame_count)
                            +'_all_mean.csv')
        
        for ddpcc_ckpt in ckpts:
            # model settings
            pcgcv2_ckpt = os.path.join(args.pcgcv2_ckpt_dir, ckpts[ddpcc_ckpt])
            ddpcc_ckpt = os.path.join(args.ckpt_dir, ddpcc_ckpt)
            checkpoint = torch.load(ddpcc_ckpt)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device).eval()
            
            # save settings per rate: tmp files
            rate = os.path.basename(ddpcc_ckpt).split('.pth')[0]
            showdir = os.path.join(BASE_DIR, test_dirname+'_'+rate)
            showdir_ = Path(showdir)
            showdir_.mkdir(exist_ok=True)
            
            csvfile = os.path.join(resultdir, seqname
                                + '_f' + str(start_frame) + '-' + str(args.frame_count)
                                +'_'+rate + '.csv')

            # test
            log_string('start testing sequence ' + seqname + ', rate point ' + ddpcc_ckpt)
            log_string('f bpp     d1PSNR  d2PSNR')

            all_frame_results = pd.DataFrame([])
            peak_value = psnr_peak_value(load_sparse_tensor(filedir_list[0], device))
            log_string('PSNR Peak value: ' + str(peak_value))
            
            # code the first frame
            if not args.f1_path:
                f1 = load_sparse_tensor(filedir_list[0], device)
                bpp, d1psnr, d2psnr, f1 = test_one_frame(f1, pcgcv2_ckpt, os.path.join(showdir, 'pcgcv2'),
                                                         res=peak_value)
                f1 = ME.SparseTensor(torch.ones_like(f1.F[:, :1]), coordinates=f1.C)
                log_string(str(0) + ' ' + str(bpp)[:7] + ' ' + str(d1psnr)[:7] + ' ' + str(d2psnr)[:7] + '\n')
                results = {'filedir': filedir_list[0], 
                           'bpp': bpp,
                           'motion_bpp': 0,
                           'residual_bpp': 0,
                           'ys2_2_lossless_bpp': 0,
                           'gpcc_bpp': 0,
                           'd1_psnr': d1psnr,
                           'd2_psnr': d2psnr,
                           'psnr_peak': peak_value,
                           'num_of_points': f1.size()[0],
                           'num_of_bits': f1.size()[0] * bpp, 
                           'motion_bits': 0,
                           'residual_bits': 0,
                           'ys2_2_lossless_bits': 0,
                           'gpcc_bits': 0,
                           'enc_time': 0,
                           'dec_time': 0, 
                           'round_motion': 0,
                           'round_residual': 0,}
                results = pd.DataFrame([results])
                all_frame_results = pd.concat([all_frame_results, results], ignore_index=True)
                all_frame_results.to_csv(csvfile, index=False)
            else:
                f1 = load_sparse_tensor(args.f1_path, device)

            # code inter frames
            for i in range(1, args.frame_count):
                if args.overwrite:
                    bitstream_filename = os.path.join(showdir, 'bitstream.bin')
                    gpcc_bitstream_filename = os.path.join(showdir, 'ys2_5.bin')
                else:
                    bitstream_filename = os.path.join(showdir, 'bitstream_' + str(i) + '.bin')
                    gpcc_bitstream_filename = os.path.join(showdir, 'ys2_5_' + str(i) + '.bin')
                
                out2, out_cls2, target2, keep2 = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
                f2 = load_sparse_tensor(filedir_list[i], device)
                num_points = f2.size()[0]

                start_time = time.time()
                enc_results = encode(f1, f2, bitstream_filename, gpcc_bitstream_filename)
                enc_time = time.time() - start_time
                log_string('Encoding frame ' + str(i) + ' time: ' + str(round(enc_time, 4)) + 's')

                ddpcc_bits = os.path.getsize(bitstream_filename) * 8
                gpcc_bits = os.path.getsize(gpcc_bitstream_filename) * 8
                bits = ddpcc_bits + gpcc_bits
                bpp = bits / num_points

                start_time = time.time()
                recon_f2_C, f2_C, recon_f2 = decode(f1, bitstream_filename, gpcc_bitstream_filename)
                dec_time = time.time() - start_time
                log_string('Decoding frame ' + str(i) + ' time: ' + str(round(dec_time, 4)) + 's')

                # D1 D2
                ori_dir, dec_dir = write_file(f2_C, recon_f2_C, showdir, i)
                PSNRs = pc_error(ori_dir, dec_dir, peak_value, normal=True)
                d1psnr = PSNRs['mseF,PSNR (p2point)'][0]
                d2psnr = PSNRs['mseF,PSNR (p2plane)'][0]
                log_string(str(i) + ' ' + str(bpp)[:7] + ' ' + str(d1psnr)[:7] + ' ' + str(d2psnr)[:7] + '\n')
                f1 = recon_f2
                
                results = {'filedir': filedir_list[i], 
                           'bpp': bpp,
                           'motion_bpp': enc_results['motion_bits'] / num_points,
                           'residual_bpp': enc_results['residual_bits'] / num_points,
                           'ys2_2_lossless_bpp': enc_results['ys2_2_lossless_bits'] / num_points,
                           'gpcc_bpp': gpcc_bits / num_points,
                           'd1_psnr': d1psnr,
                           'd2_psnr': d2psnr,
                           'psnr_peak': peak_value,
                           'num_of_points': num_points,
                           'num_of_bits': bits,
                           'motion_bits': enc_results['motion_bits'],
                           'residual_bits': enc_results['residual_bits'],
                           'ys2_2_lossless_bits': enc_results['ys2_2_lossless_bits'],
                           'gpcc_bits': gpcc_bits,
                           'enc_time': enc_time,
                           'dec_time': dec_time,
                           'round_motion': enc_results['round_motion'],
                           'round_residual': enc_results['round_residual'],}
                results = pd.DataFrame([results])
                all_frame_results = pd.concat([all_frame_results, results], ignore_index=True)
                all_frame_results.to_csv(csvfile, index=False)
                
                # TODO: optional
                show_motion(enc_results['motion_vector'], showdir, i)
                
                torch.cuda.empty_cache()
                
            # mean results of one rate
            mean_results = {'rate': rate, 
                            'bpip': all_frame_results['num_of_bits'].sum() / all_frame_results['num_of_points'].sum()}
            inter_keys = ['motion_bpp', 'residual_bpp', 'ys2_2_lossless_bpp', 'gpcc_bpp',
                            'motion_bits', 'residual_bits', 'ys2_2_lossless_bits', 'gpcc_bits',
                            'round_motion', 'round_residual']
            for col in all_frame_results.columns:
                if col in ['filedir', ]:
                    continue
                if col in inter_keys:
                    mean_results[col + '_avg'] = all_frame_results[col][1:].mean()
                else:
                    mean_results[col + '_avg'] = all_frame_results[col].mean()
            
            mean_results = pd.DataFrame([mean_results])
            all_rate_mean_results = pd.concat([all_rate_mean_results, mean_results], ignore_index=True)
            all_rate_mean_results.to_csv(all_mean_csvfile, index=False)
            log_string('Results saved to ' + csvfile)
            log_string('All rate mean results saved to ' + all_mean_csvfile)
            log_string(seqname + ' average bpp: ' + str(mean_results['bpp_avg']))
            log_string(seqname + ' average d1_psnr: ' + str(mean_results['d1_psnr_avg']))
            log_string(seqname + ' average d2_psnr: ' + str(mean_results['d2_psnr_avg']))
