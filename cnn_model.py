#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from collections import OrderedDict
import math
import copy
import torch.nn.functional as F
import sys



class DDBlock(nn.Module):
    '''
    Basic module used in Deep decoder.
    Conv(Shape invariant) -> Upsample -> ReLU -> Channel normalization
    '''
    def __init__(self, in_channel, out_channel, kernel_size):
        super(DDBlock, self).__init__()
        to_pad = kernel_size // 2
        self.dd = nn.Sequential(nn.ReflectionPad2d(to_pad),
                                nn.Conv2d(in_channel, out_channel, kernel_size),
                                nn.Upsample(scale_factor=2, mode= 'bilinear'),
                                nn.ReLU(),
                                nn.BatchNorm2d(out_channel)
                                )

    def forward(self, x):
        x = self.dd(x)
        return x


class Sum(nn.Module):
    '''
    Summation of two inputs. The one with the smaller size is upsampled, the one with fewer channels uses 0 for those
    channels, and the results have larger size and larger channels.
    '''

    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, inputs1, inputs2):
        in_data = [inputs1, inputs2]
        # check of the image size
        if (in_data[0].size(2) - in_data[1].size(2)) != 0:
            small_in_id, large_in_id = (0, 1) if in_data[0].size(2) < in_data[1].size(2) else (1, 0)
            upsample_factor = math.floor(in_data[large_in_id].size(2) / in_data[small_in_id].size(2))
            in_data[small_in_id] = F.interpolate(in_data[small_in_id], scale_factor= upsample_factor, mode= 'bilinear')

        # check of the channel size
        small_ch_id, large_ch_id = (0, 1) if in_data[0].size(1) < in_data[1].size(1) else (1, 0)
        offset = int(in_data[large_ch_id].size()[1] - in_data[small_ch_id].size()[1])
        if offset != 0:
            tmp = in_data[large_ch_id].data[:, :offset, :, :]
            tmp = Variable(tmp).clone()
            in_data[small_ch_id] = torch.cat([in_data[small_ch_id], tmp * 0], 1)
        out = torch.add(in_data[0], in_data[1])
        return out

class Concat(nn.Module):
    '''
    Concatenation of two inputs, the one with smaller size is upsampled.
    '''
    def __init__(self):
        super(Concat, self).__init__()

    def forward(self, inputs1, inputs2):
        in_data = [inputs1, inputs2]
        # check of the image size
        if (in_data[0].size(2) - in_data[1].size(2)) != 0:
            small_in_id, large_in_id = (0, 1) if in_data[0].size(2) < in_data[1].size(2) else (1, 0)
            upsample_factor = math.floor(in_data[large_in_id].size(2) / in_data[small_in_id].size(2))
            in_data[small_in_id] = F.interpolate(in_data[small_in_id], scale_factor= upsample_factor, mode= 'bilinear')
        return torch.cat([in_data[0], in_data[1]], 1)

class Out(nn.Module):
    def __init__(self, channel_num):
        super(Out, self).__init__()
        self.out = nn.Sequential(nn.Conv2d(in_channels=channel_num, out_channels=3, kernel_size=1),
                                 nn.Sigmoid()
                                 )

    def forward(self, x):
        x = self.out(x)
        return x

class CGP2CNN(nn.Module):
    def __init__(self, cgp, in_channel, imgSize):
        super(CGP2CNN, self).__init__()
        self.cgp = cgp
        self.arch = OrderedDict()
        self.encode = []
        # self.channel_num = [None for _ in range(len(self.cgp))]
        # self.size = [None for _ in range(len(self.cgp))]
        self.channel_num = [None for _ in range(500)]
        self.size = [None for _ in range(500)]
        self.channel_num[0] = in_channel
        self.size[0] = imgSize
        # encoder
        i = 0
        for name, in1, in2 in self.cgp:
            # 计算经过这个模块后的输出channel_num和size
            if name == 'input' in name:
                i += 1
                continue
            elif name == 'full':
                # 输出层是全连接
                # self.encode.append(nn.Linear(self.channel_num[in1]*self.size[in1]*self.size[in1], n_class))
                channel = self.channel_num[in1]
                self.encode.append(Out(channel))
            elif name == 'Concat':
                self.channel_num[i] = self.channel_num[in1] + self.channel_num[in2]
                small_in_id, large_in_id = (in1, in2) if self.size[in1] < self.size[in2] else (in2, in1)
                self.size[i] = self.size[large_in_id]
                self.encode.append(Concat())
            elif name == 'Sum':
                small_in_id, large_in_id = (in1, in2) if self.channel_num[in1] < self.channel_num[in2] else (in2, in1)
                self.channel_num[i] = self.channel_num[large_in_id]
                small_in_id, large_in_id = (in1, in2) if self.size[in1] < self.size[in2] else (in2, in1)
                self.size[i] = self.size[large_in_id]
                self.encode.append(Sum())
            else:
                key = name.split('_')
                down_or_up =     key[0]
                func       =     key[1]
                out_channel   = int(key[2])
                kernel     = int(key[3])
                if down_or_up == 'U':
                    if func == 'DDBlock':
                        self.channel_num[i] = out_channel
                        self.size[i] = self.size[in1]
                        self.encode.append(DDBlock(self.channel_num[in1], out_channel, kernel))
                else:
                    sys.exit('error')
                    # if func == 'ConvBlock':
                    #     self.channel_num[i] = out_size
                    #     self.size[i] = int(self.size[in1]/2)
                    #     self.encode.append(ConvBlock(self.channel_num[in1], out_size, kernel, stride=2))
                    # else:
                    #     in_data = [out_size, self.channel_num[in2]]
                    #     small_in_id, large_in_id = (in1, in2) if self.channel_num[in1] < self.channel_num[in2] else (in2, in1)
                    #     self.channel_num[i] = self.channel_num[large_in_id]
                    #     small_in_id, large_in_id = (in1, in2) if self.size[in1] < self.size[in2] else (in2, in1)
                    #     self.size[i] = self.size[small_in_id]
                    #     self.encode.append(ResBlock(self.channel_num[in1], out_size, kernel, stride=1))
            i += 1

        self.layer_module = nn.ModuleList(self.encode)
        self.outputs = [None for _ in range(len(self.cgp))]
        # 初始化结束后得到三个有用list
        # self.layer_module存储使用的模块实例（包含了具体的输入、输出大小、通道数）
        # self.channel_num存储经过对应模块后的输出通道数
        # self.size存储经过对应模块后的数据大小

    def forward(self, x):
        # 前向传播串联整个网络
        outputs = self.outputs
        outputs[0] = x  # input image
        nodeID = 1
        for layer in self.layer_module:
            if isinstance(layer, DDBlock) or isinstance(layer, Out):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]])
            elif isinstance(layer, Concat) or isinstance(layer, Sum):
                outputs[nodeID] = layer(outputs[self.cgp[nodeID][1]], outputs[self.cgp[nodeID][2]])
            else:
                sys.exit("Error at CGP2CNN forward")
            # print(layer)
            # print(outputs[nodeID].shape)
            nodeID += 1
        return outputs[nodeID - 1]
