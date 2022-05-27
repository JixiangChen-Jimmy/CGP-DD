#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Code adpated from https://github.com/sg-nm/cgp-cnn-PyTorch

@inproceedings{10.1145/3071178.3071229,
author = {Suganuma, Masanori and Shirakawa, Shinichi and Nagao, Tomoharu},
title = {A Genetic Programming Approach to Designing Convolutional Neural Network Architectures},
year = {2017},
isbn = {9781450349208},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3071178.3071229},
doi = {10.1145/3071178.3071229},
booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
pages = {497–504},
numpages = {8},
keywords = {deep learning, genetic programming, convolutional neural network, designing neural network architectures},
location = {Berlin, Germany},
series = {GECCO '17}
}
'''

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
        eval_f = CNNEvaluation(gpu_num=args.gpu_num, dataset_path='./img', verbose=True, epoch_num=500, imgSize=imgSize)

        # Execute evolution
        cgp = CGP(network_info, eval_f, lam=args.lam, imgSize=imgSize, init=args.init)
        cgp.modified_evolution(max_eval=250, mutation_rate=0.1, log_file=args.log_file)

