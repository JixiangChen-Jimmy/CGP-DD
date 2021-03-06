#!/usr/bin/env python
# -*- coding: utf-8 -*-

import multiprocessing as mp
import multiprocessing.pool
import numpy as np
import cnn_train as cnn


# wrapper function for multiprocessing
def arg_wrapper_mp(args):
    return args[0](*args[1:])

class NoDaemonProcess(mp.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NoDaemonProcessPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


# Evaluation of CNNs
def cnn_eval(net, upsample_num, gpu_id, epoch_num, dataset_path, verbose, imgSize):

    print('\tgpu_id:', gpu_id, ',', net)
    train = cnn.CNN_train(dataset_path, verbose=verbose, epoch_num= epoch_num, imgSize=imgSize)
    evaluation = train(net, upsample_num, gpu_id)
    print('\tgpu_id:', gpu_id, ', eval:', evaluation)
    return evaluation


class CNNEvaluation(object):
    def __init__(self, gpu_num, dataset_path, verbose=True, epoch_num= 500, imgSize=32):
        self.gpu_num = gpu_num
        self.dataset_path = dataset_path
        self.verbose = verbose
        self.imgSize = imgSize
        self.epoch_num = epoch_num

    def __call__(self, net_lists, upsample_num_list):
        evaluations = np.zeros(len(net_lists))
        for i in np.arange(0, len(net_lists), self.gpu_num):
            process_num = np.min((i + self.gpu_num, len(net_lists))) - i
            pool = NoDaemonProcessPool(process_num)
            arg_data = [(cnn_eval, net_lists[i+j], upsample_num_list[i+j], j, self.epoch_num, self.dataset_path, self.verbose, self.imgSize) for j in range(process_num)]
            evaluations[i:i+process_num] = pool.map(arg_wrapper_mp, arg_data)
            pool.terminate()

        return evaluations


# network configurations
class CgpInfoConvSet(object):
    '''
    Mean value code applying CGP to Deep decoder(DD) structure, I would use the basic modules in DD, which is consisted
    of conv -> Upsample -> ReLU -> Batch Normalization.
    Modules that are encoded as function/intermediate nodes in CGP vary in channels and kernel size of conv.
    '''
    def __init__(self, rows=30, cols=40, level_back=40, min_active_num=8, max_active_num=50):
        self.input_num = 1
        # ???U_??? means that the layer has a convolution layer with upsampling
        # "Sum" means that the layer has a skip connection.
        # "Concat" means that the layer has a channel connection
        self.func_type = ['U_DDBlock_8_1',  'U_DDBlock_8_3',  'U_DDBlock_8_5',
                          'U_DDBlock_16_1', 'U_DDBlock_16_3', 'U_DDBlock_16_5',
                          'U_DDBlock_32_1', 'U_DDBlock_32_3', 'U_DDBlock_32_5',
                          'Concat','Sum']
                          
        self.func_in_num = [1, 1, 1,
                            1, 1, 1,
                            1, 1, 1,
                            2, 2]

        self.out_num = 1
        self.out_type = ['full']
        self.out_in_num = [1]

        # CGP network configuration
        self.rows = rows
        self.cols = cols
        self.node_num = rows * cols
        self.level_back = level_back
        self.min_active_num = min_active_num
        self.max_active_num = max_active_num

        self.func_type_num = len(self.func_type)
        self.out_type_num = len(self.out_type)
        self.max_in_num = np.max([np.max(self.func_in_num), np.max(self.out_in_num)])
