import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

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