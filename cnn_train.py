#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from PIL import Image
from torch.autograd import Variable
import os
import copy

from cnn_model import CGP2CNN


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.apply(weights_init_normal_)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_normal_(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.uniform(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_tensor(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)

def np_to_var(img_np, dtype=torch.cuda.FloatTensor):
    '''Converts image in numpy.array to torch.Variable.

    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    return Variable(np_to_tensor(img_np)[None, :])

def var_to_np(img_var):
    '''Converts an image in torch.Variable format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.data.cpu().numpy()[0]

def get_noisy_img(img_np, sig=30, noise_same=False):
    sigma = sig / 255.
    if noise_same:  # add the same noise in each channel
        noise = np.random.normal(scale=sigma, size=img_np.shape[1:])
        noise = np.array([noise] * img_np.shape[0])
    else:  # add independent noise in each channel
        noise = np.random.normal(scale=sigma, size=img_np.shape)  # 正态分布的sigma值为30/255

    img_noisy_np = np.clip(img_np + noise, 0, 1).astype(np.float32)  #
    img_noisy_var = np_to_var(img_noisy_np).type(torch.cuda.FloatTensor)
    return img_noisy_np, img_noisy_var

def psnr(x_hat,x_true,maxv=1.):
    x_hat = x_hat.flatten()
    x_true = x_true.flatten()
    mse=np.mean(np.square(x_hat-x_true))
    psnr_ = 10.*np.log(maxv**2/mse)/np.log(10.)
    return psnr_

# __init__: load dataset
# __call__: training the CNN defined by CGP list
class CNN_train():
    def __init__(self, dataset_path, verbose=True, epoch_num = 500, imgSize=32):
        # dataset_name: name of data set ('bsds'(color) or 'bsds_gray')
        # validation: [True]  model train/validation mode
        #             [False] model test mode for final evaluation of the evolved model
        #                     (raining data : all training data, test data : all test data)
        # verbose: flag of display
        self.verbose = verbose
        self.imgSize = imgSize
        self.epoch_num = epoch_num
        self.dataset_path = dataset_path
        self.imgList = os.listdir(self.dataset_path)

    def __call__(self, cgp, upsample_num, gpuID, find_best= True, test= False):
        if self.verbose:
            print('GPUID     :', gpuID)
            print('epoch_num :', self.epoch_num)
        
        # model
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        img_psnr_list = []
        for img in self.imgList:
            img_path = self.dataset_path + '/' +img
            img_pil = Image.open(img_path)
            img_np = pil_to_np(img_pil)
            img_clean_var = np_to_var(img_np).type(torch.cuda.FloatTensor)
            img_noisy_np, img_noisy_var = get_noisy_img(img_np, sig=30, noise_same=False)

            model = CGP2CNN(cgp, 32, self.imgSize // (2 ** upsample_num))
            init_weights(model, 'kaiming')
            model.cuda(gpuID)

            # Calculate number of parameters in NN
            s = sum([np.prod(list(p.size())) for p in model.parameters()])
            print('Number of params: %d' % s)

            # Generate input of network according to the output size and the level of upsample
            width = img_clean_var.data.shape[2] // (2 ** upsample_num)
            height = img_clean_var.data.shape[3] // (2 ** upsample_num)
            shape = [1, 32, width, height]
            net_input = Variable(torch.zeros(shape))
            net_input.data.uniform_()
            net_input.data *= 1. / 10
            # Set optimizer and loss type
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            mse = torch.nn.MSELoss()

            psnr_noisy_array = []  # 储存每一代的out与噪声图像的psnr
            psnr_clear_array = []  # 储存每一代的out与真实图像的psnr

            loss_noisy_array = []  # 储存每一代的out与噪声图像的loss
            loss_clear_array = []  # 储存每一代的out与真实图像的loss

            if find_best:
                best_net = copy.deepcopy(model)
                best_mse = 1000000.0

            for i in range(self.epoch_num):

                def closure():
                    # 先将梯度归零（optimizer.zero_grad()）
                    optimizer.zero_grad()
                    out = model(net_input.type(torch.cuda.FloatTensor))

                    # training loss
                    loss = mse(out, img_noisy_var)

                    # 下面4行自己加的，用来记录psnr和loss
                    out_clone = var_to_np(out.data.clone())
                    img_noisy_var_clone = var_to_np(img_noisy_var.data.clone())
                    psnr_noisy_array.append(psnr(out_clone, img_noisy_var_clone))
                    loss_noisy_array.append(loss)

                    # 然后反向传播计算得到每个参数的梯度值（loss.backward()），
                    loss.backward()

                    # the actual loss
                    true_loss = mse(Variable(out.data, requires_grad=False), img_clean_var)
                    # 下面3行自己加的，用来记录psnr和loss
                    psnr_clear_array.append(
                        psnr(var_to_np(Variable(out.data.clone(), requires_grad=False)),
                             var_to_np(img_clean_var.data.clone())))
                    loss_clear_array.append(true_loss)

                    # 每1000次迭代print一次lss
                    if i % 100 == 0:
                        print('Iteration %05d    Train loss %f  Actual loss %f' % (
                            i, loss.data, true_loss.data), '\n', end='')
                    return loss

                loss = optimizer.step(closure)
                if find_best:
                    # if training loss improves by at least one percent, we found a new best net
                    if best_mse > 1.005 * loss.data:
                        best_mse = loss.data
                        best_psnr_clean = psnr_clear_array[-1]
            if test:
                # For retrain/test the network
                # Remain to be done
                pass

            img_psnr_list.append(best_psnr_clean)
        print('psnr list')
        print(img_psnr_list)
        return np.mean(img_psnr_list)

