"""
Original implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018
Jinwei Gu and Zhile Ren
Modified version (CMRNet) by Daniele Cattaneo
Modified version (LCCNet) by Xudong Lv
"""
import os.path
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
# from models.CMRNet.modules.attention import *
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from lietorch import SE3
from torch.autograd import Variable

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from utils import quat2mat_batch, tvector2mat_batch


# savepath = '/root/autodl-tmp/LCCNet/features'
# if not os.path.exists(savepath):
#     os.mkdir(savepath)


def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    B, C, H, W = x.shape
    for i in range(width * height):  # 图像个数
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        # plt.tight_layout() 取每一个通道的最大最小值 归一化得到特征图可视化  # B，64。H，W
        if i < C:
            img = x[0, i, :, :]  # H，W
            pmin = np.min(img)
            pmax = np.max(img)
            img = (img - pmin) / (pmax - pmin + 0.000001)
            plt.imshow(img, cmap='gray')
            print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))


# from .networks.submodules import *
# from .networks.correlation_package.correlation import Correlation
from models.correlation_package.correlation import Correlation


# __all__ = [
#     'calib_net'
# ]
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.elu = nn.ELU()
        self.leakyRELU = nn.LeakyReLU(0.1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRELU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyRELU(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leakyRELU(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, reduction=16):
        super(SEBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.leakyRELU = nn.LeakyReLU(0.1)
        # self.attention = SCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = SElayer(planes * self.expansion, ratio=reduction)
        # self.attention = ECAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = SElayer_conv(planes * self.expansion, ratio=reduction)
        # self.attention = SCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = ModifiedSCSElayer(planes * self.expansion, ratio=reduction)
        # self.attention = DPCSAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = PAlayer(planes * self.expansion, ratio=reduction)
        # self.attention = CAlayer(planes * self.expansion, ratio=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leakyRELU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.leakyRELU(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leakyRELU(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def myconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                  groups=1, bias=True),
        nn.LeakyReLU(0.1))


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225  # 归一化 B，3，256，512
        x = self.encoder.conv1(x)  # B，64，128，256
        x = self.encoder.bn1(x)  # B，64，128，256
        self.features.append(self.encoder.relu(x))  # 1，64，128，256
        # self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.maxpool(self.features[-1]))  # POOL
        self.features.append(self.encoder.layer1(self.features[-1]))  # 32，64，64，128
        self.features.append(self.encoder.layer2(self.features[-1]))  # 32，128，32，64
        self.features.append(self.encoder.layer3(self.features[-1]))  # 32，256，16，32
        self.features.append(self.encoder.layer4(self.features[-1]))  # 32，512，8，16

        return self.features  #


def bilinear_interpolate_torch(im, x, y, align=False):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    batch = x.shape[0]
    N = x.shape[1]

    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[3] - 1)
    x1 = torch.clamp(x1, 0, im.shape[3] - 1)
    y0 = torch.clamp(y0, 0, im.shape[2] - 1)
    y1 = torch.clamp(y1, 0, im.shape[2] - 1)

    x0_f = x0.reshape(-1)
    x1_f = x1.reshape(-1)
    y0_f = y0.reshape(-1)
    y1_f = y1.reshape(-1)

    B = torch.arange(0, batch).unsqueeze(-1).expand(-1, N).reshape(-1)
    Ia = im[B, :, y0_f, x0_f]
    Ib = im[B, :, y1_f, x0_f]
    Ic = im[B, :, y0_f, x1_f]
    Id = im[B, :, y1_f, x1_f]

    Ia = Ia.reshape(batch, N, -1)
    Ib = Ib.reshape(batch, N, -1)
    Ic = Ic.reshape(batch, N, -1)
    Id = Id.reshape(batch, N, -1)

    if align:
        x0 = x0.float() + 0.5
        x1 = x1.float() + 0.5
        y0 = y0.float() + 0.5
        y1 = y1.float() + 0.5

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = Ia * wa.unsqueeze(-1) + Ib * wb.unsqueeze(-1) + \
          Ic * wc.unsqueeze(-1) + Id * wd.unsqueeze(-1)
    return ans


class SparseinvariantConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        self.bias = nn.Parameter(
            torch.zeros(out_channels),
            requires_grad=True)

        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel,
            requires_grad=False)

        self.relu = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(
            kernel_size,
            stride=1,
            padding=padding)

    def forward(self, x, mask, relu=True):
        x = x * mask
        x = self.conv(x)
        normalizer = 1 / torch.clamp(self.sparsity(mask), 1)
        x = x * normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        if relu:
            x = self.relu(x)

        mask = self.max_pool(mask)

        return x, mask


class Sparseinvariantavg(nn.Module):

    def __init__(self,
                 in_channels=2,
                 kernel_size=2,
                 stride=2):
        super().__init__()

        self.sparsity = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])) \
            .unsqueeze(0).unsqueeze(0).expand(in_channels, in_channels, -1, -1)

        self.sparsity.weight = nn.Parameter(
            data=kernel,
            requires_grad=False)

        self.sparsity2 = nn.Conv2d(
            1,
            1,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity2.weight = nn.Parameter(
            data=kernel,
            requires_grad=False)

        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride)

    def forward(self, x, mask):
        x1 = x * mask
        x = self.sparsity(x1)
        x = x / torch.clamp(self.sparsity2(mask), 1)

        mask = self.max_pool(mask)

        return x, mask


class fill_hole(nn.Module):  # 不改变尺寸

    def __init__(self,
                 in_channels=2,
                 kernel_size=3):
        super().__init__()

        padding = kernel_size // 2
        self.sparsity = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            stride=(1, 1),
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])) \
            .unsqueeze(0).unsqueeze(0).expand(in_channels, in_channels, -1, -1)

        self.sparsity.weight = nn.Parameter(
            data=kernel,
            requires_grad=False)

        self.sparsity2 = nn.Conv2d(
            1,
            1,
            kernel_size=(kernel_size, kernel_size),
            padding=(padding, padding),
            stride=(1, 1),
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity2.weight = nn.Parameter(
            data=kernel,
            requires_grad=False)

        self.max_pool = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=1,
            padding=(padding, padding))

    def forward(self, x, mask):
        x1 = x * mask
        x = self.sparsity(x1)
        x = x / torch.clamp(self.sparsity2(mask), 1)
        x = (1 - mask) * x + x1  # 原先的地方不平滑
        mask = self.max_pool(mask)

        return x, mask


class LCCNet(nn.Module):
    """
    Based on the PWC-DC net. add resnet encoder, dilation convolution and densenet connections
    """

    def __init__(self, image_size, use_feat_from=1, md=4, use_reflectance=False, dropout=0.0,
                 Action_Func='leakyrelu', attention=False, res_num=18):
        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """
        super(LCCNet, self).__init__()
        input_lidar = 1
        self.res_num = res_num
        self.use_feat_from = 6
        use_feat_from = 6
        if use_reflectance:
            input_lidar = 2

        # original resnet
        self.pretrained_encoder = True
        self.net_encoder = ResnetEncoder(num_layers=self.res_num, pretrained=True, num_input_images=1)

        # resnet with leakyRELU
        self.Action_Func = Action_Func
        self.attention = attention
        if self.res_num == 50:
            layers = [3, 4, 6, 3]
            add_list = [1024, 512, 256, 64]
        elif self.res_num == 18:
            layers = [2, 2, 2, 2]
            add_list = [256, 128, 64, 64]

        if self.attention:
            block = SEBottleneck
        else:
            if self.res_num == 50:
                block = Bottleneck
            elif self.res_num == 18:
                block = BasicBlock

        # rgb_image
        self.inplanes = 64
        self.conv1_rgb = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.elu_rgb = nn.ELU()
        self.leakyRELU_rgb = nn.LeakyReLU(0.1)
        self.maxpool_rgb = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_rgb = self._make_layer(block, 64, layers[0])
        self.layer2_rgb = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_rgb = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_rgb = self._make_layer(block, 512, layers[3], stride=2)

        # lidar_image
        self.inplanes = 32
        self.spic1 = SparseinvariantConv(3, 4, 5)
        self.spic2 = SparseinvariantConv(4, 4, 5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 348 1280 -> 174 640
        self.spic3 = SparseinvariantConv(4, 16, 3)
        # 174 640 -> 128,512
        self.conv1_lidar = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 128,512->64 256
        self.elu_lidar = nn.ELU()
        self.leakyRELU_lidar = nn.LeakyReLU(0.1)
        self.maxpool_lidar = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_lidar = self._make_layer(block, 32, layers[0])
        self.layer2_lidar = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3_lidar = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4_lidar = self._make_layer(block, 256, layers[3], stride=2)

        self.reduce6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1))
        self.reduce5 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1))
        self.reduce4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1))
        self.reduce3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1))
        self.reduce2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1))

        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1,
                                corr_multiply=1)  # md=4
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])
        od = nd

        # 稠密化光流
        self.spic_flowa = fill_hole(2, 5)  # 384 1280
        self.spic_flowb = Sparseinvariantavg(2, 2, 2)  # 192 640
        self.spic_flowc = fill_hole(2, 5)  # 192 640
        self.spic_flowd = Sparseinvariantavg(2, 2, 2)  # 96 320
        self.spic_flow_down = Sparseinvariantavg(2, 2, 2)  # 48 160 # 24 80 # 12 40

        # 111111111111111
        self.conv6_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = myconv(128, 128, kernel_size=3, stride=1)
        self.conv6_2 = myconv(128, 96, kernel_size=3, stride=1)
        self.conv6_3 = myconv(96, 64, kernel_size=3, stride=1)
        self.conv6_4 = myconv(64, 32, kernel_size=3, stride=1)

        self.pos6 = nn.Conv2d(3, 32, kernel_size=1, bias=False)  # 位置编码
        self.weight6 = nn.Conv2d(32, 1, kernel_size=1, bias=False)  # 权重
        self.estimate6 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.ReLU(),
        )
        self.rot6 = nn.Linear(32, 4)
        self.trans6 = nn.Linear(32, 3)

        # 2222222222222
        od = nd + add_list[0] + 4
        self.conv5_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv5_1 = myconv(128, 128, kernel_size=3, stride=1)
        self.conv5_2 = myconv(128, 96, kernel_size=3, stride=1)
        self.conv5_3 = myconv(96, 64, kernel_size=3, stride=1)
        self.conv5_4 = myconv(64, 32, kernel_size=3, stride=1)

        self.upfeat5 = deconv(32, 2, kernel_size=4, stride=2, padding=1)

        self.pos5 = nn.Conv2d(3, 32, kernel_size=1, bias=False)  # 位置编码
        self.weight5 = nn.Conv2d(32, 1, kernel_size=1, bias=False)  # 权重
        self.estimate5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.ReLU(),
        )
        self.rot5 = nn.Linear(32, 4)
        self.trans5 = nn.Linear(32, 3)
        # 3333333333333
        od = nd + add_list[1] + 4
        self.conv4_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv4_1 = myconv(128, 128, kernel_size=3, stride=1)
        self.conv4_2 = myconv(128, 96, kernel_size=3, stride=1)
        self.conv4_3 = myconv(96, 64, kernel_size=3, stride=1)
        self.conv4_4 = myconv(64, 32, kernel_size=3, stride=1)

        self.upfeat4 = deconv(32, 2, kernel_size=4, stride=2, padding=1)

        self.pos4 = nn.Conv2d(3, 32, kernel_size=1, bias=False)  # 位置编码
        self.weight4 = nn.Conv2d(32, 1, kernel_size=1, bias=False)  # 权重
        self.estimate4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.ReLU(),
        )
        self.rot4 = nn.Linear(32, 4)
        self.trans4 = nn.Linear(32, 3)
        # 4444444444
        od = nd + add_list[2] + 4
        self.conv3_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv3_1 = myconv(128, 128, kernel_size=3, stride=1)
        self.conv3_2 = myconv(128, 96, kernel_size=3, stride=1)
        self.conv3_3 = myconv(96, 64, kernel_size=3, stride=1)
        self.conv3_4 = myconv(64, 32, kernel_size=3, stride=1)

        self.upfeat3 = deconv(32, 2, kernel_size=4, stride=2, padding=1)
        self.pos3 = nn.Conv2d(3, 32, kernel_size=1, bias=False)  # 位置编码
        self.weight3 = nn.Conv2d(32, 1, kernel_size=1, bias=False)  # 权重
        self.estimate3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.ReLU(),
        )
        self.rot3 = nn.Linear(32, 4)
        self.trans3 = nn.Linear(32, 3)

        # 55555555555
        od = nd + add_list[3] + 4
        self.conv2_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv2_1 = myconv(128, 128, kernel_size=3, stride=1)
        self.conv2_2 = myconv(128, 96, kernel_size=3, stride=1)
        self.conv2_3 = myconv(96, 64, kernel_size=3, stride=1)
        self.conv2_4 = myconv(64, 32, kernel_size=3, stride=1)
        self.upfeat2 = deconv(32, 2, kernel_size=4, stride=2, padding=1)

        self.pos2 = nn.Conv2d(3, 32, kernel_size=1, bias=False)  # 位置编码
        self.weight2 = nn.Conv2d(32, 1, kernel_size=1, bias=False)  # 权重
        self.estimate2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1, bias=False),
            nn.ReLU(),
        )
        self.rot2 = nn.Linear(32, 4)
        self.trans2 = nn.Linear(32, 3)

        self.get_weight = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=True),
            nn.Sigmoid())

        fc_size = od + dd[4]
        downsample = 128 // (2 ** use_feat_from)
        if image_size[0] % downsample == 0:
            fc_size *= image_size[0] // downsample
        else:
            fc_size *= (image_size[0] // downsample) + 1
        if image_size[1] % downsample == 0:
            fc_size *= image_size[1] // downsample
        else:
            fc_size *= (image_size[1] // downsample) + 1
        # self.fc1 = nn.Linear(fc_size * 4, 512)

        # self.fc1_trasl = nn.Linear(512, 256)
        # self.fc1_rot = nn.Linear(512, 256)
        #
        # self.fc2_trasl = nn.Linear(256, 3)
        # self.fc2_rot = nn.Linear(256, 4)

        self.dropout = nn.Dropout(dropout)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #         nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
        #         if m.bias is not None:
        #             m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo  # 光流从gt到现在

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        # mask[mask<0.9999] = 0.0
        # mask[mask>0] = 1.0
        mask = torch.floor(torch.clamp(mask, 0, 1))

        return output * mask

    def newton_gauss(self, pc, u, v, u_t, v_t, flow, weight, mask, K0):
        K = K0.float()  # flow = u_t - u
        B = flow.shape[0]
        N = flow.shape[1]
        M = 10
        # r = torch.zeros([M, B, N, 2, 1], dtype=torch.float32).cuda()  # B N 2 6
        # J = torch.zeros([M, B, N, 2, 6], dtype=torch.float32).cuda()  # B N 2 6
        # pc_f = torch.ones([B, N, 1], dtype=torch.float32).cuda()  # B N 2 6
        # pc_f = torch.cat([pc, pc_f], dim=2)  # B N 4
        pc_f = pc.clone()
        pc_f0 = pc.clone()
        fx = K[:, 0, 0:1]
        fy = K[:, 1, 1:2]
        u_t = (u + flow[:, :, 0]).float()  # 逆光流得到gt
        v_t = (v + flow[:, :, 1]).float()
        # u_t = u_t.clone().float()  # 逆光流得到gt
        # v_t = v_t.clone().float()
        T_dist = torch.zeros([B, 4, 4], dtype=torch.float32).cuda()
        T_dist[:, 0, 0] = 1
        T_dist[:, 1, 1] = 1
        T_dist[:, 2, 2] = 1
        T_dist[:, 3, 3] = 1
        W = weight.squeeze(-1).float() * mask.float()
        mask_num = mask.float().sum(dim=1)
        WH = W.unsqueeze(-1).unsqueeze(-1)
        eps = 1e-3
        for i in range(M):
            # pc_f[0]=0
            x = pc_f[:, 0, :]  # B N
            y = pc_f[:, 1, :]  # B N
            z = torch.clamp(pc_f[:, 2, :], eps, 80)  # B N
            # J = torch.zeros([B, N, 2, 6], dtype=torch.float32).cuda()  # B N 2 6
            # r = torch.zeros([B, N, 2, 1], dtype=torch.float32).cuda()  # B N 2 6如果不重复定义J和r,会有inplace覆盖问题

            J_list = []
            J_list.append(fx / z)
            J_list.append(torch.zeros([B, N], dtype=torch.float32).cuda())
            J_list.append(-fx * x / (z * z))
            J_list.append(fx * x * y / (z * z))
            J_list.append(fx + fx * x * x / (z * z))
            J_list.append(-fx * y / z)

            J_list.append(torch.zeros([B, N], dtype=torch.float32).cuda())
            J_list.append(fy / (z + eps))
            J_list.append(-fy * y / (z * z))
            J_list.append(-fy - fy * y * y / (z * z))
            J_list.append(fy * x * y / (z * z))
            J_list.append(fy * x / z)
            J = torch.stack(J_list, dim=2)
            J = J.reshape(B, N, 2, 6)
            # J[i, :, :, 0, 0] = fx / (z + eps)
            # J[i, :, :, 0, 1] = 0
            # J[i, :, :, 0, 2] = -fx * x / (z * z + eps)
            # J[i, :, :, 0, 3] = fx * x * y / (z * z + eps)
            # J[i, :, :, 0, 4] = fx + fx * x * x / (z * z + eps)
            # J[i, :, :, 0, 5] = -fx * y / (z + eps)
            # J[i, :, :, 1, 0] = 0
            # J[i, :, :, 1, 1] = fy / (z + eps)
            # J[i, :, :, 1, 2] = -fy * y / (z * z + eps)
            # J[i, :, :, 1, 3] = -fy - fy * y * y / (z * z + eps)
            # J[i, :, :, 1, 4] = fy * x * y / (z * z + eps)
            # J[i, :, :, 1, 5] = fy * x / (z + eps)
            JT = J.permute(0, 1, 3, 2)
            proj = K.matmul(pc_f[:, 0:3, :])
            u = (proj[:, 0, :] / (proj[:, 2, :] + eps))
            v = (proj[:, 1, :] / (proj[:, 2, :] + eps))
            # W = weight.squeeze(-1).float() * mask.float()  # 更新mask,这里还需要乘上网络更新的weight
            # WH = W.unsqueeze(-1).unsqueeze(-1)
            # Wb = W.unsqueeze(-1).unsqueeze(-1)

            H = JT.matmul(J)  # B N 6 6
            H = WH * H  # B N 6 6
            # H[:, :, :, :] = 0
            # a = eps * torch.eye(6).cuda()
            H = torch.mean(H, dim=1) + 1e-3 * torch.eye(6).cuda()

            # try:
            # print(H.cpu())
            # print("aaaaaaaaaaa")
            # temp = np.load('save_x.npy')
            # H = torch.from_numpy(temp)
            # H2 = H[10]
            H_1 = torch.inverse(H)
            # except:
            #     np.save('save_x',H.detach().cpu().numpy())
            #     print(H[0].cpu())
            #     print(mask_num)
            # H_1 = H_1.cuda()
            r00 = u_t - u  # B N 2 1
            r10 = v_t - v
            r = torch.stack([r00, r10], dim=2).reshape(B, N, 2, 1)
            b = WH * JT.matmul(r)  # B N 6 1
            b = b.mean(dim=1)
            dx = H_1.matmul(b).squeeze(-1)  # 得到迭代结果 # B 6
            dx = torch.clamp(dx, -10, 10)
            T_dist = SE3.exp(dx).matrix().matmul(T_dist)  # 更新 # B 4 4
            pc_f = T_dist.matmul(pc_f0)  # 恢复T_dist

            # delta_T = T_dist[0].inverse().matmul(RT)
            # print(step)
            # proj = proj.permute(0, 2, 1)
            # # step = SE3.exp(dx).matrix()
            # delta_u = err[0, :, 0, 0]
            # delta_v = err[0, :, 1, 0]
            # print('t_step:{rot}, rot_step:{t}, err{err}'
            #       .format(rot=dx[0, 0:3].norm(), t=dx[0, 3:6].norm(),
            #               err=torch.sum(torch.abs(delta_u)) + torch.sum(torch.abs(delta_v))))
        sparse_flow = K.matmul(pc_f[:, 0:3, :])
        z = sparse_flow[:, 2, :].clone()
        sparse_flow = sparse_flow[:, 0:2, :]
        sparse_flow[:, 0, :] = sparse_flow[:, 0, :].clone() / z - u
        sparse_flow[:, 1, :] = sparse_flow[:, 1, :].clone() / z - v
        return T_dist, sparse_flow

    def backbone(self, rgb, lidar):
        features1 = self.net_encoder(rgb)  # snet encode  6 feature 64，64，64，128，256，512
        c12 = features1[0]  # 2 1，64，128，256
        c13 = features1[2]  # 4 1，64，64，128
        c14 = features1[3]  # 8 1,128，32，64
        c15 = features1[4]  # 16 1，256，16，32
        c16 = features1[5]  # 32 1，512，8，16

        # lidar_image
        conv_mask = lidar[:, 2:3, :, :].bool().float()
        x2, conv_mask = self.spic1(lidar, conv_mask)
        x2, conv_mask = self.spic2(x2, conv_mask)
        x2 = self.maxpool1(x2)
        conv_mask = self.maxpool1(conv_mask)

        x2, conv_mask = self.spic3(x2, conv_mask)  # [192, 640]
        # mask_show = conv_mask.cpu().numpy()
        x2 = self.conv1_lidar(x2)  # 1，64，128，256
        if self.Action_Func == 'leakyrelu':
            c22 = self.leakyRELU_lidar(x2)  # 2
        elif self.Action_Func == 'elu':
            c22 = self.elu_lidar(x2)  # 2 1，64，128，256
        c23 = self.layer1_lidar(self.maxpool_lidar(c22))  # 4 pool 32,64,64,128
        c24 = self.layer2_lidar(c23)  # 8  32，128，32，64
        c25 = self.layer3_lidar(c24)  # 16  32，256，16，32
        c26 = self.layer4_lidar(c25)

        return c12, c13, c14, c15, c16, c22, c23, c24, c25, c26

    def get_K(self, K):
        K2 = K.clone()
        K3 = K.clone()
        K4 = K.clone()
        K5 = K.clone()
        kx2 = 256.0 / 1380.0
        ky2 = 64.0 / 384.0
        K2[:, 0, 0] = K[:, 0, 0] * kx2
        K2[:, 0, 2] = K[:, 0, 2] * kx2
        K2[:, 1, 1] = K[:, 1, 1] * ky2
        K2[:, 1, 2] = K[:, 1, 2] * ky2

        K3[:, 0, 0] = K[:, 0, 0] * kx2 / 2
        K3[:, 0, 2] = K[:, 0, 2] * kx2 / 2
        K3[:, 1, 1] = K[:, 1, 1] * ky2 / 2
        K3[:, 1, 2] = K[:, 1, 2] * ky2 / 2

        K4[:, 0, 0] = K[:, 0, 0] * kx2 / 4
        K4[:, 0, 2] = K[:, 0, 2] * kx2 / 4
        K4[:, 1, 1] = K[:, 1, 1] * ky2 / 4
        K4[:, 1, 2] = K[:, 1, 2] * ky2 / 4

        K5[:, 0, 0] = K[:, 0, 0] * kx2 / 8
        K5[:, 0, 2] = K[:, 0, 2] * kx2 / 8
        K5[:, 1, 1] = K[:, 1, 1] * ky2 / 8
        K5[:, 1, 2] = K[:, 1, 2] * ky2 / 8
        return K2, K3, K4, K5

    def densify_flow(self, uv_now, uv_new, mask0, level):
        uv_new_index = uv_new.long()  # 索引位置 B N 2 todo:量化补偿 384 1280 192 640
        B = torch.arange(0, uv_now.shape[0]).cuda().unsqueeze(-1).unsqueeze(-1).expand(-1, uv_now.shape[1], -1).long()
        uv_new_index = torch.cat([B, uv_new_index], dim=-1).view(-1, 3)  # 索引位置 BN 3
        dense_flow = torch.zeros([uv_now.shape[0], 2, 384, 1280], dtype=torch.float32).cuda()
        mask_flow = torch.zeros([uv_now.shape[0], 1, 384, 1280], dtype=torch.float32).cuda()
        mask_new = (uv_new_index[:, 1] < 1280) & (uv_new_index[:, 1] >= 0) & \
                   (uv_new_index[:, 2] < 384) & (uv_new_index[:, 2] >= 0)  # BN
        mask = mask_new * mask0.view(-1)
        uv_new_index = uv_new_index[mask, :]  # <BN  2
        delta_uv = (uv_now - uv_new).view(-1, 2)
        delta_uv = delta_uv[mask, :]
        dense_flow[uv_new_index[:, 0], :, uv_new_index[:, 2], uv_new_index[:, 1]] = delta_uv
        mask_flow[uv_new_index[:, 0], :, uv_new_index[:, 2], uv_new_index[:, 1]] = 1

        dense_flow, mask_flow = self.spic_flowa(dense_flow, mask_flow)
        dense_flow, mask_flow = self.spic_flowb(dense_flow, mask_flow)
        dense_flow, mask_flow = self.spic_flowc(dense_flow, mask_flow)
        dense_flow, mask_flow = self.spic_flowd(dense_flow, mask_flow)

        mask_new = mask_new.view(uv_now.shape[0], -1).unsqueeze(1).unsqueeze(-1)

        num = mask_new.float().squeeze().sum(dim=1)

        if (level == 5):
            dense_flow, mask_flow = self.spic_flow_down(dense_flow, mask_flow)
            dense_flow, mask_flow = self.spic_flow_down(dense_flow, mask_flow)
            dense_flow, mask_flow = self.spic_flow_down(dense_flow, mask_flow)
            # print(num)
            return dense_flow, mask_new
        if (level == 4):
            dense_flow, mask_flow = self.spic_flow_down(dense_flow, mask_flow)
            dense_flow, mask_flow = self.spic_flow_down(dense_flow, mask_flow)
            return dense_flow, mask_new
        if (level == 3):
            dense_flow, mask_flow = self.spic_flow_down(dense_flow, mask_flow)
            return dense_flow, mask_new
        return dense_flow, mask_new

    def forward(self, rgb, lidar, uv, pcl_xyz, mask, K):  # B，N，H  插值
        # rgb_image
        # 1.主干网络
        # 2.初始化参数 mask
        # 3.warp,推理q和t
        # 4.计算这个层次的光流
        # 5.判断点的个数,点少于5个则判断此次迭代失败
        K = K.float()
        uv = uv.float()
        c12, c13, c14, c15, c16, c22, c23, c24, c25, c26 = self.backbone(rgb, lidar)

        # 准备参数
        ignore = (~mask).view(-1)  # 用于pointweight
        mask_points = mask.unsqueeze(1).unsqueeze(-1)
        valid_list = torch.zeros(uv.shape[0], 5, dtype=torch.bool).cuda()  # 如果一次迭代以后,小于10个点在相机视野内,后面的迭代loss置零
        # K2, K3, K4, K5 = self.get_K(K)

        # 第一次迭代 384 1280  192 640  ||  96 320  48 160  24 80  12 40  6 20
        corr6 = self.corr(self.reduce6(c16), c26)
        corr6 = self.leakyRELU(corr6)
        x = self.conv6_0(corr6)
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        x = self.conv6_3(x)
        x = self.conv6_4(x)
        # num = mask.float().sum(dim=1)
        point_feature6 = bilinear_interpolate_torch(x, uv[:, :, 0] / 64, uv[:, :, 1] / 64).float()
        pcl_xyz6 = pcl_xyz.clone()
        xyz = pcl_xyz6[:, 0:3, :].unsqueeze(-1)
        pos_feature6 = self.pos6(xyz)
        point_feature6 = pos_feature6 + point_feature6.permute(0, 2, 1).unsqueeze(-1)
        point_feature6 = self.estimate6(point_feature6)
        point_weight6 = self.weight6(point_feature6).view(-1)

        point_weight6[ignore] = -1e9
        point_weight6 = point_weight6.view(point_feature6.shape[0], 1, point_feature6.shape[2], 1)
        softmax = nn.Softmax(dim=2)
        point_weight6 = softmax(point_weight6) * mask_points
        point_feature6 = (point_feature6 * point_weight6).sum(dim=2).squeeze(-1)
        q6 = self.leakyRELU(self.rot6(point_feature6))
        q6 = F.normalize(q6, dim=1)
        t6 = self.leakyRELU(self.trans6(point_feature6))  # 从gt到现在

        R_target6 = quat2mat_batch(q6)
        T_target6 = tvector2mat_batch(t6)
        RT_target6 = torch.bmm(T_target6, R_target6)
        pcl_xyz5 = torch.bmm(RT_target6.inverse(), pcl_xyz6)  # 现在到gt

        uv5 = torch.bmm(K, pcl_xyz5[:, 0:3, :])  # 新的uv
        # loss = uv5.sum()
        # loss.backward()
        # temp = uv5[:, 2, :].clone()
        temp = uv5[:, 2, :].clone()
        temp = torch.clamp(temp, 1e-3)
        uv5[:, 0, :] = uv5[:, 0, :].clone() / temp
        uv5[:, 1, :] = uv5[:, 1, :].clone() / temp
        uv5 = uv5[:, 0:2, :].permute(0, 2, 1)

        # 注意这个光流是在384 1280尺度上的
        # loss = uv5.sum()
        # loss.backward()
        # loss = uv5.sum()
        # uv5 = uv5[:, 0:2, :].permute(0, 2, 1)
        # loss = uv5.sum()
        # loss.backward()
        dense_flow5, mask5 = self.densify_flow(uv, uv5, mask, 5)  # mask5 是新点还在内部的掩码
        valid_iter = (mask5.int().squeeze().sum(dim=1) > 5)
        valid_list[:, 0] = True
        valid_list[:, 1] = valid_iter

        # 第二次迭代
        warp5 = self.warp(c25, dense_flow5 / 32)
        corr5 = self.corr(self.reduce5(c15), warp5)
        corr5 = self.leakyRELU(corr5)
        up_feat5 = self.upfeat5(x)
        x = torch.cat((corr5, c15, dense_flow5, up_feat5), 1)
        x = self.conv5_0(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)

        point_feature5 = bilinear_interpolate_torch(x, uv5[:, :, 0] / 32, uv5[:, :, 1] / 32).float()

        xyz = pcl_xyz5[:, 0:3, :].unsqueeze(-1)
        pos_feature5 = self.pos5(xyz)
        point_feature5 = pos_feature5 + point_feature5.permute(0, 2, 1).unsqueeze(-1)
        point_feature5 = self.estimate5(point_feature5)
        point_weight5 = self.weight5(point_feature5).view(-1)

        point_weight5[ignore] = -1e9
        point_weight5 = point_weight5.view(point_feature5.shape[0], 1, point_feature5.shape[2], 1)
        softmax = nn.Softmax(dim=2)
        point_weight5 = softmax(point_weight5) * mask_points * mask5
        point_feature5 = (point_feature5 * point_weight5).sum(dim=2).squeeze(-1)
        q5 = self.rot5(point_feature5)
        q5 = F.normalize(q5, dim=1)
        t5 = self.trans5(point_feature5)  # 从gt到现在

        R_target5 = quat2mat_batch(q5)
        T_target5 = tvector2mat_batch(t5)
        RT_target5 = torch.bmm(T_target5, R_target5)
        pcl_xyz4 = torch.bmm(RT_target5.inverse(), pcl_xyz5)  # 现在到gt

        uv4 = torch.bmm(K, pcl_xyz4[:, 0:3, :])  # 新的uv
        temp = torch.clamp(uv4[:, 2, :].clone(), 1e-3)
        uv4[:, 0, :] = uv4[:, 0, :].clone() / temp
        uv4[:, 1, :] = uv4[:, 1, :].clone() / temp
        # 注意这个光流是在384 1280尺度上的
        uv4 = uv4[:, 0:2, :].permute(0, 2, 1)
        dense_flow4, mask4 = self.densify_flow(uv, uv4, mask, 4)
        valid_iter = (mask4.int().squeeze().sum(dim=1) > 5)
        valid_list[:, 2] = valid_iter * valid_list[:, 1]  # 前一次迭代失败,后面都视为失败

        # 第三次迭代
        up_feat4 = self.upfeat4(x)
        warp4 = self.warp(c24, dense_flow4 / 16)
        corr4 = self.corr(self.reduce4(c14), warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, dense_flow4, up_feat4), 1)
        x = self.conv4_0(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)

        point_feature4 = bilinear_interpolate_torch(x, uv4[:, :, 0] / 16, uv4[:, :, 1] / 16).float()

        xyz = pcl_xyz4[:, 0:3, :].unsqueeze(-1)
        pos_feature4 = self.pos4(xyz)
        point_feature4 = pos_feature4 + point_feature4.permute(0, 2, 1).unsqueeze(-1)
        point_feature4 = self.estimate4(point_feature4)
        point_weight4 = self.weight4(point_feature4).view(-1)

        point_weight4[ignore] = -1e9
        point_weight4 = point_weight4.view(point_feature4.shape[0], 1, point_feature4.shape[2], 1)
        softmax = nn.Softmax(dim=2)
        point_weight4 = softmax(point_weight4) * mask_points * mask4
        point_feature4 = (point_feature4 * point_weight4).sum(dim=2).squeeze(-1)
        q4 = self.rot4(point_feature4)
        q4 = F.normalize(q4, dim=1)
        t4 = self.trans4(point_feature4)  # 从gt到现在

        R_target4 = quat2mat_batch(q4)
        T_target4 = tvector2mat_batch(t4)
        RT_target4 = torch.bmm(T_target4, R_target4)
        pcl_xyz3 = torch.bmm(RT_target4.inverse(), pcl_xyz4)  # 现在到gt

        uv3 = torch.bmm(K, pcl_xyz3[:, 0:3, :])  # 新的uv
        temp = torch.clamp(uv3[:, 2, :].clone(), 1e-3)
        uv3[:, 0, :] = uv3[:, 0, :].clone() / temp
        uv3[:, 1, :] = uv3[:, 1, :].clone() / temp
        # 注意这个光流是在384 1280尺度上的
        uv3 = uv3[:, 0:2, :].permute(0, 2, 1)
        dense_flow3, mask3 = self.densify_flow(uv, uv3, mask, 3)
        valid_iter = (mask3.int().squeeze().sum(dim=1) > 5)
        valid_list[:, 3] = valid_iter * valid_list[:, 2]

        # 第四次迭代
        up_feat3 = self.upfeat3(x)
        warp3 = self.warp(c23, dense_flow3 / 8)
        corr3 = self.corr(self.reduce3(c13), warp3)
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((corr3, c13, dense_flow3, up_feat3), 1)
        x = self.conv3_0(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        point_feature3 = bilinear_interpolate_torch(x, uv3[:, :, 0] / 8, uv3[:, :, 1] / 8).float()

        xyz = pcl_xyz3[:, 0:3, :].unsqueeze(-1)
        pos_feature3 = self.pos3(xyz)
        point_feature3 = pos_feature3 + point_feature3.permute(0, 2, 1).unsqueeze(-1)
        point_feature3 = self.estimate3(point_feature3)
        point_weight3 = self.weight3(point_feature3).view(-1)

        point_weight3[ignore] = -1e9
        point_weight3 = point_weight3.view(point_feature3.shape[0], 1, point_feature3.shape[2], 1)
        softmax = nn.Softmax(dim=2)
        point_weight3 = softmax(point_weight3) * mask_points * mask3
        point_feature3 = (point_feature3 * point_weight3).sum(dim=2).squeeze(-1)
        q3 = self.rot3(point_feature3)
        q3 = F.normalize(q3, dim=1)
        t3 = self.trans3(point_feature3)  # 从gt到现在

        R_target2 = quat2mat_batch(q3)
        T_target2 = tvector2mat_batch(t3)
        RT_target2 = torch.bmm(T_target2, R_target2)
        pcl_xyz2 = torch.bmm(RT_target2.inverse(), pcl_xyz3)  # 现在到gt
        uv2 = torch.bmm(K, pcl_xyz2[:, 0:3, :])  # 新的uv
        temp = torch.clamp(uv2[:, 2, :].clone(), 1e-3)
        uv2[:, 0, :] = uv2[:, 0, :].clone() / temp
        uv2[:, 1, :] = uv2[:, 1, :].clone() / temp
        uv2 = uv2[:, 0:2, :].permute(0, 2, 1)
        dense_flow2, mask2 = self.densify_flow(uv, uv2, mask, 2)
        valid_iter = (mask2.int().squeeze().sum(dim=1) > 5)
        valid_list[:, 4] = valid_iter * valid_list[:, 3]

        # 第五次迭代
        up_feat2 = self.upfeat2(x)
        warp2 = self.warp(c22, dense_flow2 / 4)
        corr2 = self.corr(self.reduce2(c12), warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, dense_flow2, up_feat2), 1)
        x = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv2_4(x)
        point_feature2 = bilinear_interpolate_torch(x, uv2[:, :, 0] / 4, uv2[:, :, 1] / 4).float()

        xyz = pcl_xyz2[:, 0:3, :].unsqueeze(-1)
        pos_feature2 = self.pos2(xyz)
        point_feature2 = pos_feature2 + point_feature2.permute(0, 2, 1).unsqueeze(-1)
        point_feature2 = self.estimate2(point_feature2)
        point_weight2 = self.weight2(point_feature2).view(-1)

        point_weight2[ignore] = -1e9
        point_weight2 = point_weight2.view(point_feature2.shape[0], 1, point_feature2.shape[2], 1)
        softmax = nn.Softmax(dim=2)
        point_weight2 = softmax(point_weight2) * mask_points * mask2
        point_feature2 = (point_feature2 * point_weight2).sum(dim=2).squeeze(-1)
        q2 = self.rot2(point_feature2)
        q2 = F.normalize(q2, dim=1)
        t2 = self.trans2(point_feature2)  # 从gt到现在

        return q6, t6, q5, t5, q4, t4, q3, t3, q2, t2, valid_list  # transl, rot
