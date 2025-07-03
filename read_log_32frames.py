import numpy as np
from pathlib import Path
import open3d
import torch
import torch.utils.data as data
from os.path import join
import os
import MinkowskiEngine as ME
import random
import pandas as pd

root_dir = os.path.dirname(os.path.abspath(__file__))
log_names = ['96frames-0.txt', '96frames-1.txt', '96frames-2.txt', '96frames-3.txt']
seq_names = ['basketball', 'dancer', 'exercise', 'model']
results = {
    'basketball': {'bpp': [], 'd1-psnr': [], 'd2-psnr': []},
    'dancer': {'bpp': [], 'd1-psnr': [], 'd2-psnr': []},
    'exercise': {'bpp': [], 'd1-psnr': [], 'd2-psnr': []},
    'model': {'bpp': [], 'd1-psnr': [], 'd2-psnr': []}
}
results_dir = './mpeg-results-32-10bit'
results_dir_ = Path(results_dir)
results_dir_.mkdir(exist_ok=True)

for log_name, seq_name in zip(log_names, seq_names):
    log_dir = os.path.join(root_dir, log_name)
    reader = open(log_dir, 'r')
    flag = 0
    count = 0

    for line in reader:
        words = line.split()
        # print(words)
        if seq_name+',' in words and './final_ckpts/new-I2_6.pth' in words:
            flag = 1
            bpp_sum = 0
            d1_psnr_sum, d2_psnr_sum = 0, 0
        if flag == 1:
            if len(words) >= 7 and words[5] == 'INFO' and words[3] == 'Model' and words[7].isdigit():
                if 0 <= int(words[7]) < 32:
                    count += 1
                    # print(int(words[7]))
                    # print(line)
                    bpp_sum += float(words[8])
                    d1_psnr_sum += float(words[9])
                    d2_psnr_sum += float(words[10])
                    if int(words[7]) == 31:
                        results[seq_name]['bpp'].append(bpp_sum/count)
                        results[seq_name]['d1-psnr'].append(d1_psnr_sum/count)
                        results[seq_name]['d2-psnr'].append(d2_psnr_sum/count)
                        flag = 0
                        count = 0

        if seq_name+',' in words and './final_ckpts/new-I4-5-1.pth' in words:
            flag = 2
            bpp_sum = 0
            d1_psnr_sum, d2_psnr_sum = 0, 0
        if flag == 2:
            if len(words) >= 7 and words[5] == 'INFO' and words[3] == 'Model' and words[7].isdigit():
                if 0 <= int(words[7]) < 32:
                    count += 1
                    # print(int(words[7]))
                    # print(line)
                    bpp_sum += float(words[8])
                    d1_psnr_sum += float(words[9])
                    d2_psnr_sum += float(words[10])
                    if int(words[7]) == 31:
                        results[seq_name]['bpp'].append(bpp_sum/count)
                        results[seq_name]['d1-psnr'].append(d1_psnr_sum/count)
                        results[seq_name]['d2-psnr'].append(d2_psnr_sum/count)
                        flag = 0
                        count = 0

        if seq_name+',' in words and './final_ckpts/new-I5-15.pth' in words:
            flag = 3
            bpp_sum = 0
            d1_psnr_sum, d2_psnr_sum = 0, 0
        if flag == 3:
            if len(words) >= 7 and words[5] == 'INFO' and words[3] == 'Model' and words[7].isdigit():
                if 0 <= int(words[7]) < 32:
                    count += 1
                    # print(int(words[7]))
                    # print(line)
                    bpp_sum += float(words[8])
                    d1_psnr_sum += float(words[9])
                    d2_psnr_sum += float(words[10])
                    if int(words[7]) == 31:
                        results[seq_name]['bpp'].append(bpp_sum/count)
                        results[seq_name]['d1-psnr'].append(d1_psnr_sum/count)
                        results[seq_name]['d2-psnr'].append(d2_psnr_sum/count)
                        flag = 0
                        count = 0

        if seq_name+',' in words and './final_ckpts/new-I7-15.pth' in words:
            flag = 4
            bpp_sum = 0
            d1_psnr_sum, d2_psnr_sum = 0, 0
        if flag == 4:
            if len(words) >= 7 and words[5] == 'INFO' and words[3] == 'Model' and words[7].isdigit():
                if 0 <= int(words[7]) < 32:
                    count += 1
                    # print(int(words[7]))
                    # print(line)
                    bpp_sum += float(words[8])
                    d1_psnr_sum += float(words[9])
                    d2_psnr_sum += float(words[10])
                    if int(words[7]) == 31:
                        results[seq_name]['bpp'].append(bpp_sum/count)
                        results[seq_name]['d1-psnr'].append(d1_psnr_sum/count)
                        results[seq_name]['d2-psnr'].append(d2_psnr_sum/count)
                        flag = 0
                        count = 0

        if seq_name+',' in words and './final_ckpts/new-I9.pth' in words:
            flag = 5
            bpp_sum = 0
            d1_psnr_sum, d2_psnr_sum = 0, 0
        if flag == 5:
            if len(words) >= 7 and words[5] == 'INFO' and words[3] == 'Model' and words[7].isdigit():
                if 0 <= int(words[7]) < 32:
                    count += 1
                    # print(int(words[7]))
                    bpp_sum += float(words[8])
                    d1_psnr_sum += float(words[9])
                    d2_psnr_sum += float(words[10])
                    if int(words[7]) == 31:
                        results[seq_name]['bpp'].append(bpp_sum/count)
                        results[seq_name]['d1-psnr'].append(d1_psnr_sum/count)
                        results[seq_name]['d2-psnr'].append(d2_psnr_sum/count)
                        flag = 0
                        count = 0

        if seq_name+',' in words and './final_ckpts/new-I11.pth' in words:
            flag = 6
            bpp_sum = 0
            d1_psnr_sum, d2_psnr_sum = 0, 0
        if flag == 6:
            if len(words) >= 7 and words[5] == 'INFO' and words[3] == 'Model' and words[7].isdigit():
                if 0 <= int(words[7]) < 32:
                    count += 1
                    # print(int(words[7]))
                    bpp_sum += float(words[8])
                    d1_psnr_sum += float(words[9])
                    d2_psnr_sum += float(words[10])
                    if int(words[7]) == 31:
                        results[seq_name]['bpp'].append(bpp_sum/count)
                        results[seq_name]['d1-psnr'].append(d1_psnr_sum/count)
                        results[seq_name]['d2-psnr'].append(d2_psnr_sum/count)
                        flag = 0
                        count = 0

        # if seq_name+',' in words and './final_ckpts/new-I12-150.pth' in words:
        #     flag = 7
        #     bpp_sum = 0
        #     d1_psnr_sum, d2_psnr_sum = 0, 0
        # if flag == 7:
        #     if len(words) >= 7 and words[5] == 'INFO' and words[3] == 'Model' and words[7].isdigit():
        #         if 0 <= int(words[7]) < 32:
        #             count += 1
        #             # print(int(words[7]))
        #             bpp_sum += float(words[8])
        #             d1_psnr_sum += float(words[9])
        #             d2_psnr_sum += float(words[10])
        #             if int(words[7]) == 31:
        #                 results[seq_name]['bpp'].append(bpp_sum/count)
        #                 results[seq_name]['d1-psnr'].append(d1_psnr_sum/count)
        #                 results[seq_name]['d2-psnr'].append(d2_psnr_sum/count)
        #                 flag = 0
        #                 count = 0

        if seq_name+',' in words and './final_ckpts/new-I15.pth' in words:
            flag = 8
            bpp_sum = 0
            d1_psnr_sum, d2_psnr_sum = 0, 0
        if flag == 8:
            if len(words) >= 7 and words[5] == 'INFO' and words[3] == 'Model' and words[7].isdigit():
                if 0 <= int(words[7]) < 32:
                    count += 1
                    # print(int(words[7]))
                    bpp_sum += float(words[8])
                    d1_psnr_sum += float(words[9])
                    d2_psnr_sum += float(words[10])
                    if int(words[7]) == 31:
                        results[seq_name]['bpp'].append(bpp_sum/count)
                        results[seq_name]['d1-psnr'].append(d1_psnr_sum/count)
                        results[seq_name]['d2-psnr'].append(d2_psnr_sum/count)
                        flag = 0
                        count = 0

    df = pd.DataFrame(results[seq_name])
    df.to_csv(os.path.join(results_dir, seq_name + '-32frames.csv'), index=False)
