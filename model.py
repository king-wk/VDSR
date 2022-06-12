import random
import torch
import torch.nn as nn
from math import sqrt


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        # # 一共 20 层卷积层
        # self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)

        # 第一层处理输入图像
        # 输入的图像是低分辨率图像插值之后的图像
        # 输入图像的通道是1，只输入了 Y 通道
        self.input = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 修改模型，以便跳出
        self.residual_layer = nn.ModuleList()
        for _ in range(18):
            modules_body = [Conv_ReLU_Block()]
            self.residual_layer.append(nn.Sequential(*modules_body))
        
        # 最后一层图像重建
        self.output = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x, idx=None):
        residual = x
        out = self.relu(self.input(x))

        assert idx <= 18 and idx >= 0, "output_node is invalid"
        
        for i in range(idx):
            out = self.residual_layer[i](out)
        
        out = self.output(out)
        out = torch.add(out, residual)
        return out
