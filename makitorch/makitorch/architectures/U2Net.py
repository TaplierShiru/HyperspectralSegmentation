import torch
from torch import nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, in_f, out_f, k_size=3, dialation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_f, out_channels=out_f, kernel_size=k_size, padding=dialation, dilation=dialation)
        self.bn = nn.BatchNorm2d(num_features=out_f)
    
    def forward(self, x):
        out = self.bn(self.conv(x))
        out = torch.relu(out)
        return out


class DownsampleBlock(nn.Module):
    def __init__(self, in_f, out_f, k_size=3, dialation=1):
        super().__init__()
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.block = ConvBnRelu(in_f, out_f, k_size, dialation)
    
    def forward(self, x):
        hx = F.max_pool2d(x, kernel_size=2, stride=2)
        out = self.block(hx)
        return out


def upsampling(source, target):
    x = F.interpolate(source, size=target.shape[2:], mode='bilinear')
    return x


class RSU(nn.Module):
    def __init__(self, in_f, out_f, M, N):
        """
        in_f - C_in from original paper. Number of input features
        out_f - C_ou from original paper. Number of output features
        M - M from original paper. Number of channels in internal layers
        N - Number of up/down layers in each encoder and decoder parts
        """
        super().__init__()
        self.convBlock1 = ConvBnRelu(in_f, out_f)
        self.convBlock2 = ConvBnRelu(out_f, M)

        self.downPath = nn.ModuleList([DownsampleBlock(M, M) for i in range(N)])

        self.middleConvBlock = ConvBnRelu(M, M, dialation=2)

        self.upPath = nn.ModuleList([ConvBnRelu(2 * M, M) for i in range(N)] + [ConvBnRelu(2 * M, out_f)])

    def forward(self, x):
        layer_outputs = []
        layer_outputs.append(self.convBlock1(x))
        layer_outputs.append(self.convBlock2(layer_outputs[-1]))

        for i, l in enumerate(self.downPath):
            layer_outputs.append(l(layer_outputs[-1]))

        x = self.middleConvBlock(layer_outputs[-1])

        for i, l in enumerate(self.upPath):
            concatted_layer = layer_outputs[-1 - i]
            in_layer = torch.cat((x, concatted_layer), dim=1)
            x = l(in_layer)
            x = upsampling(x, layer_outputs[-2 - i])

        return layer_outputs[0] + x


class RSUF(nn.Module):
    def __init__(self, in_f, out_f, M):
        """
        in_f - C_in from original paper. Number of input features
        out_f - C_ou from original paper. Number of output features
        M - M from original paper. Number of channels in internal layers
        """
        super().__init__()
        self.conv1_d1 = ConvBnRelu(in_f, out_f, dialation=1)
        self.conv2_d1 = ConvBnRelu(out_f, M, dialation=1)

        self.conv1_d2 = ConvBnRelu(M, M, dialation=2)
        self.conv1_d4 = ConvBnRelu(M, M, dialation=4)
        self.conv_d8 = ConvBnRelu(M, M, dialation=8)

        self.conv2_d4 = ConvBnRelu(M * 2, M, dialation=4)
        self.conv2_d2 = ConvBnRelu(M * 2, M, dialation=2)
        self.conv3_d1 = ConvBnRelu(M * 2, out_f, dialation=1)
        

    def forward(self, x):
        hxin = self.conv1_d1(x)

        hx1 = self.conv2_d1(hxin)
        hx2 = self.conv1_d2(hx1)
        hx3 = self.conv1_d4(hx2)

        hx4 = self.conv_d8(hx3)

        hx3d = self.conv2_d4(torch.cat((hx4,hx3),1))
        hx2d = self.conv2_d2(torch.cat((hx3d,hx2),1))
        hx1d = self.conv3_d1(torch.cat((hx2d,hx1),1))

        return hx1d + hxin


class U2Net(nn.Module):
    def __init__(self, in_f=3, out_f=1):
        super().__init__()
        self.stage1 = RSU(in_f=in_f, out_f=64, M=32, N=5)
        self.stage2 = RSU(64, 128, 32, 4)
        self.stage3 = RSU(128, 256, 64, 3)
        self.stage4 = RSU(256, 512, 128, 2)
        self.stage5 = RSUF(512, 512, 256)

        self.stage6 = RSUF(512, 512, 256)
        
        self.stageDe5 = RSUF(1024, 512, 256)
        self.stageDe4 = RSU(1024, 256, 128, 2)
        self.stageDe3 = RSU(512, 128, 64, 3)
        self.stageDe2 = RSU(256, 64, 32, 4)
        self.stageDe1 = RSU(128, 64, 16, 5)
        
        self.side6 = nn.Conv2d(512, out_f, kernel_size=3, padding=1)
        self.side5 = nn.Conv2d(512, out_f, kernel_size=3, padding=1)
        self.side4 = nn.Conv2d(256, out_f, kernel_size=3, padding=1)
        self.side3 = nn.Conv2d(128, out_f, kernel_size=3, padding=1)
        self.side2 = nn.Conv2d(64, out_f, kernel_size=3, padding=1)
        self.side1 = nn.Conv2d(64, out_f, kernel_size=3, padding=1)

        self.outConv = nn.Conv2d(6 * out_f, out_f, kernel_size=3, padding=1)

    def forward(self, x):
        hx1 = self.stage1(x)
        hx = F.max_pool2d(hx1, kernel_size=2, stride=2)

        hx2 = self.stage2(hx)
        hx = F.max_pool2d(hx2, kernel_size=2, stride=2)

        hx3 = self.stage3(hx)
        hx = F.max_pool2d(hx3, kernel_size=2, stride=2)

        hx4 = self.stage4(hx)
        hx = F.max_pool2d(hx4, kernel_size=2, stride=2)

        hx5 = self.stage5(hx)
        hx = F.max_pool2d(hx5, kernel_size=2, stride=2)

        hx6 = self.stage6(hx)
        hx = upsampling(hx6, hx5)

        hx5up = self.stageDe5(torch.cat((hx, hx5), dim=1))
        hx = upsampling(hx5up, hx4)

        hx4up = self.stageDe4(torch.cat((hx, hx4), dim=1))
        hx = upsampling(hx4up, hx3)

        hx3up = self.stageDe3(torch.cat((hx, hx3), dim=1))
        hx = upsampling(hx3up, hx2)

        hx2up = self.stageDe2(torch.cat((hx, hx2), dim=1))
        hx = upsampling(hx2up, hx1)

        hx1up = self.stageDe1(torch.cat((hx, hx1), dim=1))

        side1 = self.side1(hx1up)
        
        
        side2 = upsampling(self.side2(hx2up), side1)
        side3 = upsampling(self.side3(hx3up), side1)
        side4 = upsampling(self.side4(hx4up), side1)
        side5 = upsampling(self.side5(hx5up), side1)
        side6 = upsampling(self.side6(hx6), side1)

        out = self.outConv(torch.cat((side1, side2, side3, side4, side5, side6), dim=1))

        return F.sigmoid(out), F.sigmoid(side1), F.sigmoid(side2), F.sigmoid(side3), F.sigmoid(side4), F.sigmoid(side5), F.sigmoid(side6)
