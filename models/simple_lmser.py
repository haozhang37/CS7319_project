import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import os
import numpy as np


class SimpleLmser(nn.Module):
    def __init__(self, class_num=10, reflect_num=3, layer_num=3, channel=128):
        super(SimpleLmser, self).__init__()
        self.fc = nn.ModuleList([])
        self.dec_fc = nn.ModuleList([])
        self.layer_num = layer_num
        self.reflect = reflect_num
        for i in range(self.layer_num):
            in_c, out_c = channel, channel
            if i == 0:
                in_c = 28 * 28
            if i == self.layer_num - 1:
                out_c = class_num
            self.fc.append(nn.Linear(in_c, out_c, bias=False))
            self.dec_fc.append(nn.Linear(out_c, in_c, bias=False))

        self.set_DCW()

    def set_DCW(self):
        for i in range(self.layer_num):
            self.fc[i].weight.data = (self.fc[i].weight + self.dec_fc[i].weight.transpose(0, 1)) / 2
            self.dec_fc[i].weight.data = self.fc[i].weight.data.transpose(0, 1)

    def forward(self, x):
        recurrent = [0 for _ in range(self.layer_num)]
        for i in range(self.reflect):
            short_cut = []
            for j in range(self.layer_num):
                x = F.sigmoid(self.fc[j](x + recurrent[j]))
                short_cut.append(x)
            recurrent = []
            for j in range(self.layer_num):
                l = self.layer_num - j - 1
                x = F.sigmoid(self.dec_fc[l](x + short_cut[l]))
                recurrent.append(x)
            recurrent = recurrent[::-1]
        return x


if __name__ == '__main__':
    x = torch.randn((10, 28 * 28))
    model = SimpleLmser(class_num=10)
    y = model(x)
    z = y.sum()
    z.backward()
    print(x.size())
    print(y.size())
