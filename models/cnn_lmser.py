import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import os
import numpy as np


class CNNLmser(nn.Module):
    def __init__(self, class_num=10, reflect_num=1, block_num=3, layer_num=2, channel=None, img_channel=3, img_size=96):
        super(CNNLmser, self).__init__()
        if channel is None:
            channel = [32 * (2 ** i) for i in range(block_num)]
        self.conv = nn.ModuleList([])
        self.dec_conv = nn.ModuleList([])
        self.layer_num = layer_num
        self.reflect = reflect_num
        self.block_num = block_num
        self.channel = channel
        for i in range(self.block_num):
            for j in range(self.layer_num):
                in_c, out_c = channel[i], channel[i]
                k = 3
                stride, padding = 1, 1
                if i == 0 and j == 0:
                    in_c = img_channel
                if j == self.layer_num - 1 and i != self.block_num - 1:
                    out_c = channel[i + 1]
                    stride, padding = 2, 0
                    k = 2
                elif j == self.layer_num - 1 and i == self.block_num - 1:
                    out_c = class_num
                    stride, padding = 1, 0
                    k = int(img_size)
                self.conv.append(nn.Conv2d(in_c, out_c, kernel_size=k, stride=stride, padding=padding, bias=False))
                self.dec_conv.append(nn.ConvTranspose2d(out_c, in_c, kernel_size=k, stride=stride, padding=padding, bias=False))
            img_size /= 2
        self.set_DCW()

    def set_DCW(self):
        for i in range(self.layer_num):
            self.conv[i].weight.data = (self.conv[i].weight + self.dec_conv[i].weight) / 2
            self.dec_conv[i].weight.data = self.conv[i].weight.data

    def forward(self, x):
        recurrent = [0 for _ in range(self.layer_num * self.block_num)]
        for i in range(self.reflect):
            short_cut = []
            for j in range(self.block_num):
                for k in range(self.layer_num):
                    l = j * self.layer_num + k
                    if k != self.layer_num - 1 and j != self.block_num - 1:
                        x = F.sigmoid(self.conv[l](x + recurrent[l]))
                    else:
                        x = self.conv[l](x + recurrent[l])
                    short_cut.append(x)
            recurrent = []
            for j in range(self.block_num):
                for k in range(self.layer_num):
                    l = self.layer_num * self.block_num - j * self.layer_num - k - 1
                    if j != self.block_num - 1 and k != self.layer_num:
                        x = F.sigmoid(self.dec_conv[l](x + short_cut[l]))
                    else:
                        x = self.dec_conv[l](x + short_cut[l])
                    recurrent.append(x)
            recurrent = recurrent[::-1]
        return x


if __name__ == '__main__':
    x = torch.randn((4, 3, 96, 96))
    model = CNNLmser()
    # test_con = nn.Conv2d(3, 64, kernel_size=2, stride=2, bias=False)
    # test_dconv = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2, bias=False)
    # test_dconv.weight.data = test_con.weight.data
    # a = test_con(x)
    # b = test_dconv(a)
    y = model(x)
    print()
