"""
    Compared with new_test_owlii_mpeg.py
    use mpeg_dmetric to calculate PSNR rather than manual function.
"""
import argparse
import importlib
import logging
import sys
import os


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Test Script')
    parser.add_argument('--model', type=str, default='DDPCC_geo')
    parser.add_argument('--lossless_model', type=str, default='DDPCC_lossless_coder')
    parser.add_argument('--log_name', type=str, default='bbb')
    parser.add_argument('--gpu', type=str, default='4', help='specify gpu device [default: 0]')
    parser.add_argument('--channels', default=8, type=int)
    parser.add_argument('--ckpt_dir', type=str,
                        default='./final_ckpts')
    parser.add_argument('--pcgcv2_ckpt_dir', type=str,
                        default='./pcgcv2_ckpts')
    parser.add_argument('--frame_count', type=int, default=100, help='number of frames to be coded')
    parser.add_argument('--results_dir', type=str, default='mpeg-results-pcerror', help='directory to store results (in csv format)')
    parser.add_argument('--tmp_dir', type=str, default='tmp')
    parser.add_argument('--overwrite', type=bool, default=False, help='overwrite the bitstream of previous frame')
    parser.add_argument('--dataset_dir', type=str, default='/home/old/gaolinyao/Owlii_10bit')
    return parser.parse_args()
args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, './PCGCv2'))

from models.model_utils import index_points, sort_by_coor_sum, coordinate_sort_by_coor_sum, quant
from dataset_owlii import *
from models.entropy_coding import *
from GPCC.gpcc_wrapper import *
from PCGCv2.eval import test_one_frame
import pandas as pd
import collections, math
from pytorch3d.ops import knn_points


def log_string(string):
    logger.info(string)
    print(string)


def psnr_peak_value(x: torch.Tensor):
    if isinstance(x, torch.Tensor):
        coordinates = x.detach().cpu().numpy()
    elif isinstance(x, ME.SparseTensor):
        coordinates = x.C[:, 1:].detach().cpu().numpy()
    else:
        raise TypeError("Input must be torch.Tensor or ME.SparseTensor")
    resolution = 2 ** np.round(np.log2(coordinates.max() - coordinates.min())) - 1
    return int(resolution)


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
    write_ply_data(os.path.join(tmp_dir, 'ys2_5.ply'), ys2_5_C / 8)
    gpcc_encode(os.path.join(tmp_dir, 'ys2_5.ply'), gpcc_bitstream_filename)
    file = open(bitstream_filename, 'wb')
    file.write(np.array(min_v_motion_1, dtype=np.int8).tobytes())
    file.write(np.array(max_v_motion_1, dtype=np.int8).tobytes())
    file.write(np.array(min_v_motion_2, dtype=np.int8).tobytes())
    file.write(np.array(max_v_motion_2, dtype=np.int8).tobytes())
    file.write(np.array(min_v_res, dtype=np.int8).tobytes())
    file.write(np.array(max_v_res, dtype=np.int8).tobytes())
    file.write(np.array(min_v_res2, dtype=np.int8).tobytes())
    file.write(np.array(max_v_res2, dtype=np.int8).tobytes())
    file.write(np.array(quant_y.shape[1], dtype=np.uint16).tobytes())
    file.write(np.array(quant_motion_1.shape[0], dtype=np.uint16).tobytes())
    file.write(np.array(quant_motion_2.shape[0], dtype=np.uint16).tobytes())
    file.write(np.array(ys2[0].shape[0], dtype=np.uint32).tobytes())
    file.write(np.array(ys2[1].shape[0], dtype=np.uint32).tobytes())
    file.write(np.array(len(motion_bitstream_1), dtype=np.uint16).tobytes())
    file.write(np.array(len(motion_bitstream_2), dtype=np.uint16).tobytes())
    file.write(np.array(len(ys2_2_feature_bitstream), dtype=np.uint16).tobytes())
    file.write(np.array(len(ys2_2_bitstream), dtype=np.uint16).tobytes())
    file.write(motion_bitstream_1)
    file.write(motion_bitstream_2)
    file.write(ys2_2_feature_bitstream)
    file.write(ys2_2_bitstream)
    file.write(residual_bitstream)
    file.close()


def decode(f1, bitstream_filename, gpcc_bitstream_filename):
    ys1 = [f1, 0, 0, 0]
    file = open(bitstream_filename, 'rb')
    min_v_motion_1_, max_v_motion_1_, min_v_motion_2_, max_v_motion_2_, min_v_res_, max_v_res_, min_v_res2_, max_v_res2_ \
        = np.frombuffer(file.read(8), dtype=np.int8)
    quant_y_length, quant_motion_1_length, quant_motion_2_length = np.frombuffer(
        file.read(6), dtype=np.uint16)
    num_points_0, num_points_1 = np.frombuffer(
        file.read(8), dtype=np.uint32)
    motion_bitstream_1_length, motion_bitstream_2_length, ys2_2_feature_bitstream_length, ys2_2_bitstream_length = np.frombuffer(
        file.read(8), dtype=np.uint16)
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

    gpcc_decode(gpcc_bitstream_filename, os.path.join(tmp_dir, 'recon_ys2_5.ply'))
    recon_ys2_5_C = 8 * torch.tensor(read_point_cloud(os.path.join(tmp_dir, 'recon_ys2_5.ply')),
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


if __name__ == '__main__':
    device = torch.device('cuda')
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('./%s.txt' % args.log_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    tmp_dir = args.tmp_dir
    # tmp_dir = './tmp_'+''.join(random.sample('0123456789', 10))
    tmp_dir_ = Path(tmp_dir)
    tmp_dir_.mkdir(exist_ok=True)
    results_dir = args.results_dir
    results_dir_ = Path(results_dir)
    results_dir_.mkdir(exist_ok=True)
    gpcc_bitstream_filename = os.path.join(tmp_dir, 'gpcc.bin')

    # load model
    log_string('PARAMETER ...')
    log_string(args)
    MODEL = importlib.import_module(args.model)
    model = MODEL.get_model(channels=args.channels)
    model.eval()

    LOSSLESS_MODEL = importlib.import_module(args.lossless_model)
    lossless_model = LOSSLESS_MODEL.get_model()
    lossless_checkpoint = torch.load('./final_ckpts/lossless_coder.pth')
    old_paras = lossless_model.state_dict()
    new_state_dict = collections.OrderedDict()
    for k, v in lossless_checkpoint['model_state_dict'].items():
        k1 = k.replace('module.', '')
        if k1 in old_paras:
            new_state_dict[k1] = v
    old_paras.update(new_state_dict)
    lossless_model.load_state_dict(old_paras)
    lossless_model = lossless_model.to(device).eval()

    results = {
        'basketball': {'bpp': [], 'd1-psnr': [], 'd2-psnr': [], 'exp_name': []},
        'dancer': {'bpp': [], 'd1-psnr': [], 'd2-psnr': [], 'exp_name': []},
        'exercise': {'bpp': [], 'd1-psnr': [], 'd2-psnr': [], 'exp_name': []},
        'model': {'bpp': [], 'd1-psnr': [], 'd2-psnr': [], 'exp_name': []}
    }
    '''
    start testing
    0: basketballplayer
    1: dancer
    2: exercise
    3: model
    '''
    ckpts = {
        'r4_0.15bpp.pth': 'new-I2_6.pth',
        'r4_0.15bpp-1.pth': 'new-I4-5-1.pth',
        'r4_0.15bpp-2.pth': 'new-I5-15.pth',
        'r5_0.25bpp.pth': 'new-I7-15.pth',
        'r6_0.3bpp.pth': 'new-I9.pth',
        'r6_0.3bpp-1.pth': 'new-I11.pth',
        # 'r7_0.4bpp.pth': 'new-I12-150.pth',
        'r7_0.4bpp-1.pth': 'new-I15.pth',
    }
    with torch.no_grad():
        for pcgcv2_ckpt in ckpts:
            exp_name = str(ckpts[pcgcv2_ckpt]).split('.')[0]
            ddpcc_ckpt = os.path.join(args.ckpt_dir, ckpts[pcgcv2_ckpt])
            pcgcv2_ckpt = os.path.join(args.pcgcv2_ckpt_dir, pcgcv2_ckpt)
            checkpoint = torch.load(ddpcc_ckpt, map_location='cuda:0')
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device).eval()
            for sequence in (0, 1, 2, 3, ):
                dataset = Dataset(root_dir=args.dataset_dir, split=[sequence], type='test', format='ply')
                sequence_name = dataset.sequence_list[sequence]
                log_string('start testing sequence ' + sequence_name + ', rate point ' + ddpcc_ckpt)
                log_string('f bpp     d1PSNR  d2PSNR')
                d1_psnr_sum = 0
                d2_psnr_sum = 0
                bpp_sum = 0
                bits_sum = 0
                num_points_sum = 0
                enc_time_sum = 0
                dec_time_sum = 0

                # encode the first frame
                xyz, point, xyz1, point1 = collate_pointcloud_fn([dataset[0]])
                f1 = ME.SparseTensor(features=point, coordinates=xyz, device=device)
                peak_value = psnr_peak_value(f1)
                print('peak_value:', peak_value)
                bpp, d1psnr, d2psnr, f1 = test_one_frame(f1, pcgcv2_ckpt, os.path.join(tmp_dir, 'pcgcv2'), res=peak_value+1)
                f1 = ME.SparseTensor(torch.ones_like(f1.F[:, :1]), coordinates=f1.C)
                log_string(str(0) + ' ' + str(bpp)[:7] + ' ' + str(d1psnr)[:7] + ' ' + str(d2psnr)[:7] + '\n')
                bpp_sum += bpp
                d1_psnr_sum += d1psnr
                d2_psnr_sum += d2psnr
                num_points_sum += (f1.size()[0] * 1.0)
                bits_sum += (f1.size()[0] * bpp)

                for i in range(1, args.frame_count):
                    if args.overwrite:
                        bitstream_filename = os.path.join(tmp_dir, 'bitstream.bin')
                        gpcc_bitstream_filename = os.path.join(tmp_dir, 'ys2_5.bin')
                    else:
                        bitstream_filename = os.path.join(tmp_dir, 'bitstream_' + str(i) + '.bin')
                        gpcc_bitstream_filename = os.path.join(tmp_dir, 'ys2_5_' + str(i) + '.bin')

                    out2, out_cls2, target2, keep2 = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
                    xyz, point, xyz1, point1 = collate_pointcloud_fn([dataset[i-1]])
                    f1 = sort_by_coor_sum(f1)
                    f2 = ME.SparseTensor(features=point1, coordinates=xyz1, device=device)
                    num_points = f2.size()[0]

                    encoding_st = time.time()
                    encode(f1, f2, bitstream_filename, gpcc_bitstream_filename)
                    encoding_et = time.time()
                    log_string('encoding time: ' + str(encoding_et - encoding_st))
                    ddpcc_bpp = os.path.getsize(bitstream_filename) * 8 / num_points
                    gpcc_bpp = os.path.getsize(gpcc_bitstream_filename) * 8 / num_points
                    bpp = ddpcc_bpp + gpcc_bpp

                    decoding_st = time.time()
                    recon_f2_C, f2_C, recon_f2 = decode(f1, bitstream_filename, gpcc_bitstream_filename)
                    decoding_et = time.time()
                    log_string('decoding time: ' + str(decoding_et - decoding_st))

                    if i != 1:
                        enc_time_sum += encoding_et - encoding_st
                        dec_time_sum += decoding_et - decoding_st

                    # D1 D2
                    # write_ply_data(os.path.join(tmp_dir, 'f2.ply'), f2_C)
                    write_ply_open3d_normal(os.path.join(tmp_dir, 'f2.ply'), f2_C)
                    write_ply_data(os.path.join(tmp_dir, 'f2_recon.ply'), recon_f2_C)
                    # print('psnr resolution\t', args.resolution - 1)
                    PSNRs = pc_error(os.path.join(tmp_dir, 'f2.ply'), os.path.join(tmp_dir, 'f2_recon.ply'), res=peak_value+1,
                                     normal=True)
                    d1psnr = PSNRs['mseF,PSNR (p2point)'][0]
                    d2psnr = PSNRs['mseF,PSNR (p2plane)'][0]
                    log_string(str(i) + ' ' + str(bpp)[:7] + ' ' + str(d1psnr)[:7] + ' ' + str(d2psnr)[:7] + '\n')
                    f1 = recon_f2
                    bpp_sum += bpp
                    d1_psnr_sum += d1psnr
                    d2_psnr_sum += d2psnr
                    num_points_sum += (num_points * 1.0)

                # sequence results
                bpp_avg = bpp_sum / args.frame_count
                d1_psnr_avg = d1_psnr_sum / args.frame_count
                d2_psnr_avg = d2_psnr_sum / args.frame_count
                results[sequence_name]['bpp'].append(bpp_avg)
                results[sequence_name]['d1-psnr'].append(d1_psnr_avg)
                results[sequence_name]['d2-psnr'].append(d2_psnr_avg)
                results[sequence_name]['exp_name'].append(exp_name)
                log_string(dataset.sequence_list[sequence] + ' average bpp: ' + str(bpp_avg))
                log_string(dataset.sequence_list[sequence] + ' average d1-psnr: ' + str(d1_psnr_avg))
                log_string(dataset.sequence_list[sequence] + ' average d2-psnr: ' + str(d2_psnr_avg))
                log_string('average_enc_time: ' + str(enc_time_sum / (args.frame_count - 2)))
                log_string('average_dec_time: ' + str(dec_time_sum / (args.frame_count - 2)))

                # save sequence results
                df = pd.DataFrame(results[sequence_name])
                df.to_csv(os.path.join(results_dir, sequence_name + '.csv'), index=False)
