#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Code adpated from https://github.com/reinhardh/supplement_deep_decoder

@article{heckel_deep_2018,
    author    = {Reinhard Heckel and Paul Hand},
    title     = {Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks},
    journal   = {International Conference on Learning Representations},
    year      = {2019}
}
'''

import numpy as np
import torch
from torch.nn import init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from PIL import Image
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt

from cnn_model import CGP2CNN

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

def plotfig(array1, array2,typep,img_name, param):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 6))

    array1 = np.array(array1)
    array2 = np.array(array2)

    x = np.arange(len(array1))

    axes[0].plot(x, array1, 'g', linewidth=2)
    axes[0].set_ylabel('noisy', fontsize=17)

    axes[1].plot(x, array2, 'g', linewidth=2, linestyle="--")
    axes[1].set_ylabel('clean', fontsize=17)

    plt.suptitle(f'{img_name}_{typep}_plot, param_{param}')
    # plt.show()
    fig.savefig(f'./test_result/{img_name}/{img_name}_{typep}_plot.png')

def myimgshow(plt, img):
    plt.imshow(np.clip(img.transpose(1, 2, 0), 0, 1))

def plot_results(out_img_np, img_np, img_noisy_np, iteration,net_parameter, img_name):
    fig = plt.figure(figsize=(15, 8))  # create a 5 x 5 figure

    ax1 = fig.add_subplot(131)
    myimgshow(ax1, img_np)
    ax1.set_title('Original image')
    ax1.axis('off')

    ax2 = fig.add_subplot(132)
    myimgshow(ax2, img_noisy_np)
    ax2.set_title("Noisy observation, PSNR: %.2f" % psnr(img_np, img_noisy_np))
    ax2.axis('off')

    ax3 = fig.add_subplot(133)
    myimgshow(ax3, out_img_np)
    ax3.set_title("DD denoised image, PSNR: %.2f" % psnr(img_np, out_img_np))
    ax3.axis('off')
    fig.suptitle(f'{img_name} compare result, epoch= {iteration}, param= {net_parameter}')
    # plt.show()
    fig.savefig(f'./test_result/{img_name}/{img_name}_visulization_results.png')

# __init__: load dataset
# __call__: training the CNN defined by CGP list
class CNN_train():
    def __init__(self, dataset_path, verbose=True, epoch_num = 500, imgSize=32):
        # verbose: flag of display
        self.verbose = verbose
        self.imgSize = imgSize
        self.epoch_num = epoch_num
        self.dataset_path = dataset_path
        self.imgList = os.listdir(self.dataset_path)

    def __call__(self, cgp, upsample_num, gpuID, test= False):
        if self.verbose:
            print('GPUID     :', gpuID)
            print('epoch_num :', self.epoch_num)
        
        # model
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        img_psnr_list = []
        for img in self.imgList:
            img_path = self.dataset_path + '/' +img
            img_name_split = img.split('.')
            img_name = img_name_split[0]
            img_pil = Image.open(img_path)
            img_np = pil_to_np(img_pil)
            img_clean_var = np_to_var(img_np).type(torch.cuda.FloatTensor)
            img_noisy_np, img_noisy_var = get_noisy_img(img_np, sig=30, noise_same=False)

            model = CGP2CNN(cgp, 32, self.imgSize // (2 ** upsample_num))
            # init_weights(model, 'kaiming')
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
                    if i % 10 == 0:
                        print('Iteration %05d    Train loss %f  Actual loss %f' % (
                            i, loss.data, true_loss.data), '\n', end='')
                        print(f'psnr = {psnr_clear_array[-1]}')
                    return loss

                loss = optimizer.step(closure)
            if test:
                if not os.path.exists('./test_result'):
                    os.makedirs('./test_result')
                if not os.path.exists(f'./test_result/{img_name}'):
                    os.makedirs(f'./test_result/{img_name}')
                # Test a searched network, set the input mode as 'test' to activate this
                # Draw the clean image, noisy image, recovered image
                # Plot the loss funciton curve, psnr curve
                plotfig(psnr_noisy_array, psnr_clear_array, 'psnr', img_name, s)
                plotfig(loss_noisy_array, loss_clear_array, 'loss', img_name, s)
                out_img_np = model(net_input.type(torch.cuda.FloatTensor)).data.cpu().numpy()[0]
                plot_results(out_img_np, img_np, img_noisy_np, self.epoch_num, s, img_name)
            img_psnr_list.append(psnr_clear_array[-1])

        print('psnr list')
        print(img_psnr_list)
        return np.mean(img_psnr_list)

