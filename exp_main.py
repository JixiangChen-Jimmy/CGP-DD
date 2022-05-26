#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pickle
# import pandas as pd

from cgp import *
from cgp_config import *
from cnn_train import CNN_train


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evolving CAE structures')
    parser.add_argument('--gpu_num', '-g', type=int, default=1, help='Num. of GPUs')
    parser.add_argument('--lam', '-l', type=int, default=2, help='Num. of offsprings')
    parser.add_argument('--net_info_file', default='network_info.pickle', help='Network information file name')
    parser.add_argument('--log_file', default='./log_cgp.txt', help='Log file name')
    parser.add_argument('--mode', '-m', default='evolution', help='Mode (evolution / retrain / reevolution)')
    parser.add_argument('--init', '-i', action='store_true')
    args = parser.parse_args()

    # --- Optimization of the CNN architecture ---
    if args.mode == 'evolution':
        # Create CGP configuration and save network information
        network_info = CgpInfoConvSet(rows=5, cols=30, level_back=10, min_active_num=3, max_active_num=30)
        # 将Python数据转换并保存到pickle格式的文件内，写入的文件是二进制文件，直接打开不可读
        with open(args.net_info_file, mode='wb') as f:
            pickle.dump(network_info, f)
        # Evaluation function for CGP (testing unlearned NN and return average PSNR)
        imgSize = 256
        eval_f = CNNEvaluation(gpu_num=args.gpu_num, dataset_path='./img', verbose=True, epoch_num=3000, imgSize=imgSize)

        # Execute evolution
        cgp = CGP(network_info, eval_f, lam=args.lam, imgSize=imgSize, init=args.init)
        cgp.modified_evolution(max_eval=250, mutation_rate=0.1, log_file=args.log_file)

