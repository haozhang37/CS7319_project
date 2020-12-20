import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import os
import numpy as np


class SimpleLmser(nn.Module):
    def __init__(self, class_num=10, reflect_num=3, layer_num=3, channel=128, img_size=28):
        super(SimpleLmser, self).__init__()
        self.fc = nn.ModuleList([])
        self.dec_fc = nn.ModuleList([])
        self.layer_num = layer_num
        self.reflect = reflect_num
        for i in range(self.layer_num):
            in_c, out_c = channel, channel
            if i == 0:
                in_c = img_size * img_size
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
                if j != self.layer_num - 1:
                    x = F.sigmoid(self.fc[j](x + recurrent[j]))
                else:
                    x = self.fc[j](x + recurrent[j])
                short_cut.append(x)
            recurrent = []
            for j in range(self.layer_num):
                l = self.layer_num - j - 1
                if j != self.layer_num - 1:
                    x = F.sigmoid(self.dec_fc[l](x + short_cut[l]))
                else:
                    x = self.dec_fc[l](x + short_cut[l])
                recurrent.append(x)
            recurrent = recurrent[::-1]
        return x


class Pse_Inv_Lmser(nn.Module):
    def __init__(self, class_num=10, reflect_num=3, layer_num=3, channel=128, img_size=28):
        super(Pse_Inv_Lmser, self).__init__()
        self.fc = nn.ModuleList([])
        self.dec_fc = nn.ModuleList([])
        self.layer_num = layer_num
        self.reflect = reflect_num
        for i in range(self.layer_num):
            in_c, out_c = channel, channel
            if i == 0:
                in_c = img_size * img_size
            if i == self.layer_num - 1:
                out_c = class_num
            self.fc.append(nn.Linear(in_c, out_c, bias=False))
            self.dec_fc.append(nn.Linear(out_c, in_c, bias=False))

        self.set_DPN()

    def set_DPN(self):
        for i in range(self.layer_num):
            weight = np.array(self.fc[i].weight.data.cpu())
            dec_weight = np.linalg.pinv(weight)
            self.dec_fc[i].weight.data = torch.from_numpy(dec_weight).to(self.fc[i].weight.data.device)

    def forward(self, x):
        recurrent = [0 for _ in range(self.layer_num)]
        for i in range(self.reflect):
            short_cut = []
            for j in range(self.layer_num):
                # x = self.fc[j](x + recurrent[j])
                if j != self.layer_num - 1:
                    # x = F.relu(self.fc[j](x + recurrent[j]))
                    x = F.sigmoid(self.fc[j](x + recurrent[j]))
                else:
                    x = self.fc[j](x + recurrent[j])
                short_cut.append(x)
            recurrent = []
            for j in range(self.layer_num):
                l = self.layer_num - j - 1
                if j != self.layer_num - 1:
                    # x = self.dec_fc[l](x + short_cut[l])
                    # x = F.relu(self.dec_fc[l](x + short_cut[l]))
                    x = F.sigmoid(self.dec_fc[l](x + short_cut[l]))
                else:
                    x = self.dec_fc[l](x + short_cut[l])
                recurrent.append(x)
            recurrent = recurrent[::-1]
        return x


if __name__ == '__main__':
    x = torch.randn((10, 28 * 28))
    model = Pse_Inv_Lmser(class_num=10, reflect_num=1)
    y = model(x)
    z = y.sum()
    z.backward()
    print(x.size())
    print(y.size())
