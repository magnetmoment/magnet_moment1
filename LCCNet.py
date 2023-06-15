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

    def forward(self, x, mask):
        x = x * mask
        x = self.conv(x)
        normalizer = 1 / torch.clamp(self.sparsity(mask), 1)
        x = x * normalizer + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        x = self.relu(x)

        mask = self.max_pool(mask)

        return x, mask


class Sparseinvariantavg(nn.Module):

    def __init__(self,
                 in_channels=2,
                 out_channels=2,
                 kernel_size=2,
                 stride=2):
        super().__init__()

        padding = kernel_size - 1 // 2

        self.sparsity = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False)

        kernel = torch.FloatTensor(torch.ones([kernel_size, kernel_size])).unsqueeze(0).unsqueeze(0)

        self.sparsity.weight = nn.Parameter(
            data=kernel,
            requires_grad=False)

        self.max_pool = nn.MaxPool2d(
            kernel_size,
            stride=1,
            padding=padding)

    def forward(self, x, mask):
        x1 = x * mask
        x = self.sparsity(x1)
        x = x / torch.clamp(self.sparsity(mask), 1)

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
        self.spic1 = SparseinvariantConv(1, 4, 5)
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
        self.conv6_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = myconv(128, 128, kernel_size=3, stride=1)
        self.conv6_2 = myconv(128, 96, kernel_size=3, stride=1)
        self.conv6_3 = myconv(96, 64, kernel_size=3, stride=1)
        self.conv6_4 = myconv(64, 32, kernel_size=3, stride=1)

        # 稠密化光流
        self.spic_flow1 = SparseinvariantConv(2, 2, 5)  # 128 512
        self.spic_flow2 = SparseinvariantConv(2, 2, 5)  # 128 512
        self.spic_flow3 = Sparseinvariantavg(2, 2, 2, 2)  # 64 256
        self.spic_flow4 = Sparseinvariantavg(2, 2, 2, 2)  # 32 128
        self.spic_flow5 = Sparseinvariantavg(2, 2, 2, 2)  # 16 64
        self.spic_flow6 = Sparseinvariantavg(2, 2, 2, 2)  # 8 32

        if use_feat_from > 1:
            self.upfeat6 = deconv(32, 2, kernel_size=2, stride=2)

            self.pos6 = nn.Conv2d(3, 32, kernel_size=1, bias=False)  # 位置编码
            self.weight6 = nn.Conv2d(32, 1, kernel_size=1, bias=False)  # 权重
            self.estimate6 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, bias=False),
                nn.ReLU(),
            )
            self.rot6 = nn.Linear(32, 4)
            self.trans6 = nn.Linear(32, 3)

            od = nd + add_list[0] + 4
            self.conv5_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv5_1 = myconv(128, 128, kernel_size=3, stride=1)
            self.conv5_2 = myconv(128, 96, kernel_size=3, stride=1)
            self.conv5_3 = myconv(96, 64, kernel_size=3, stride=1)
            self.conv5_4 = myconv(64, 32, kernel_size=3, stride=1)

        if use_feat_from > 2:
            self.upfeat5 = deconv(32, 2, kernel_size=2, stride=2)

            self.pos5 = nn.Conv2d(3, 32, kernel_size=1, bias=False)  # 位置编码
            self.weight5 = nn.Conv2d(32, 1, kernel_size=1, bias=False)  # 权重
            self.estimate5 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, bias=False),
                nn.ReLU(),
            )
            self.rot5 = nn.Conv2d(32, 4, kernel_size=1, bias=False),
            self.trans5 = nn.Conv2d(32, 3, kernel_size=1, bias=False),

            od = nd + add_list[1] + 4
            self.conv4_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv4_1 = myconv(128, 128, kernel_size=3, stride=1)
            self.conv4_2 = myconv(128, 96, kernel_size=3, stride=1)
            self.conv4_3 = myconv(96, 64, kernel_size=3, stride=1)
            self.conv4_4 = myconv(64, 32, kernel_size=3, stride=1)

        if use_feat_from > 3:
            self.upfeat4 = deconv(32, 2, kernel_size=2, stride=2)

            self.pos4 = nn.Conv2d(3, 32, kernel_size=1, bias=False)  # 位置编码
            self.weight4 = nn.Conv2d(32, 1, kernel_size=1, bias=False)  # 权重
            self.estimate4 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, bias=False),
                nn.ReLU(),
            )
            self.rot4 = nn.Conv2d(32, 4, kernel_size=1, bias=False),
            self.trans4 = nn.Conv2d(32, 3, kernel_size=1, bias=False),

            od = nd + add_list[2] + 4
            self.conv3_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv3_1 = myconv(128, 128, kernel_size=3, stride=1)
            self.conv3_2 = myconv(128, 96, kernel_size=3, stride=1)
            self.conv3_3 = myconv(96, 64, kernel_size=3, stride=1)
            self.conv3_4 = myconv(64, 32, kernel_size=3, stride=1)

        if use_feat_from > 4:
            self.upfeat3 = deconv(32, 2, kernel_size=2, stride=2)
            self.pos3 = nn.Conv2d(3, 32, kernel_size=1, bias=False)  # 位置编码
            self.weight3 = nn.Conv2d(32, 1, kernel_size=1, bias=False)  # 权重
            self.estimate3 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, bias=False),
                nn.ReLU(),
            )
            self.rot3 = nn.Conv2d(32, 4, kernel_size=1, bias=False),
            self.trans3 = nn.Conv2d(32, 3, kernel_size=1, bias=False),

            od = nd + add_list[3] + 4
            self.conv2_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv2_1 = myconv(128, 128, kernel_size=3, stride=1)
            self.conv2_2 = myconv(128, 96, kernel_size=3, stride=1)
            self.conv2_3 = myconv(96, 64, kernel_size=3, stride=1)
            self.conv2_4 = myconv(64, 32, kernel_size=3, stride=1)

            self.pos2 = nn.Conv2d(3, 32, kernel_size=1, bias=False)  # 位置编码
            self.weight2 = nn.Conv2d(32, 1, kernel_size=1, bias=False)  # 权重
            self.estimate2 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, bias=False),
                nn.ReLU(),
            )
            self.rot2 = nn.Conv2d(32, 4, kernel_size=1, bias=False),
            self.trans2 = nn.Conv2d(32, 3, kernel_size=1, bias=False),

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

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()

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
        conv_mask = lidar.bool().float()
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

    def densify_flow(self, uv_now, uv_new, mask, level):
        uv_now_index = uv_now.long()  # 索引位置 B 2 N  todo:量化补偿 384 1280 192 640
        dense_flow = torch.zeros([uv_now.shape[0], 2, 384, 1280], dtype=torch.float32).cuda()
        mask_flow = torch.zeros([uv_now.shape[0], 1, 384, 1280], dtype=torch.float32).cuda()
        mask = (uv_now_index[:, 1, :] < 1280) & (uv_now_index[:, 1, :] >= 0) & \
               (uv_now_index[:, 0, :] < 384) & (uv_now_index[:, 0, :] >= 0)
        dense_flow

    def forward(self, rgb, lidar, uv, uv_t, pcl_xyz, mask, K):  # B，N，H  插值
        # rgb_image
        # 1.主干网络
        # 2.初始化参数 mask和每个层次的内参
        # 3.推理q和t
        # 4.计算这个层次的光流
        # 5.warp
        K = K.float()
        c12, c13, c14, c15, c16, c22, c23, c24, c25, c26 = self.backbone(rgb, lidar)

        # 准备参数
        ignore = (~mask).view(-1)
        # K2, K3, K4, K5 = self.get_K(K)

        # 第一次迭代 384 1280  192 640  ||  96 320  48 160  24 80  12 40  6 20
        corr6 = self.corr(self.reduce6(c16), c26)
        corr6 = self.leakyRELU(corr6)
        x = self.conv6_0(corr6)
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        x = self.conv6_3(x)
        x = self.conv6_4(x)
        point_feature = bilinear_interpolate_torch(x, uv[:, :, 0] / 64, uv[:, :, 1] / 64).float()
        pcl_xyz6 = pcl_xyz.clone()
        xyz = pcl_xyz6[:, 0:3, :].unsqueeze(-1)
        pos_feature = self.pos6(xyz)
        point_feature = pos_feature + point_feature.permute(0, 2, 1).unsqueeze(-1)
        point_feature = self.estimate6(point_feature)
        point_weight = self.weight6(point_feature).view(-1)

        point_weight[ignore] = -1e9
        point_weight = point_weight.view(point_feature.shape[0], 1, point_feature.shape[2], 1)
        softmax = nn.Softmax(dim=2)
        point_weight = softmax(point_weight)
        point_feature = (point_feature * point_weight).sum(dim=2).squeeze(-1)
        q6 = self.rot6(point_feature)
        q6 = F.normalize(q6, dim=1)
        t6 = self.trans6(point_feature)  # 从gt到现在

        R_target = quat2mat_batch(q6)
        T_target = tvector2mat_batch(t6)
        RT_target = torch.bmm(T_target, R_target)
        pcl_xyz5 = torch.bmm(RT_target.inverse(), pcl_xyz6)  # 现在到gt

        uv5 = torch.bmm(K, pcl_xyz5[:, 0:3, :])  # 新的uv
        uv5[:, 0, :] = uv5[:, 0, :] / uv5[:, 2, :]
        uv5[:, 1, :] = uv5[:, 1, :] / uv5[:, 2, :]
        # 注意这个光流是在384 1280尺度上的
        self.densify_flow(uv, uv5[:,0:2,:].permute(0, 2, 1), mask, 6)

        # 第二次迭代
        warp5 = self.warp(c25, delta_uv5)
        corr5 = self.corr(self.reduce5(c15), warp5)
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = self.conv5_0(x)
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        sparse_flow3 = bilinear_interpolate_torch(up_flow3, uv[:, :, 0] * kx4, uv[:, :, 1] * ky4).float()
        sparse_flow4 = bilinear_interpolate_torch(up_flow4, uv[:, :, 0] * kx4 / 2, uv[:, :, 1] * ky4 / 2).float()
        sparse_flow5 = bilinear_interpolate_torch(up_flow5, uv[:, :, 0] * kx4 / 4, uv[:, :, 1] * ky4 / 4).float()
        sparse_flow6 = bilinear_interpolate_torch(up_flow6, uv[:, :, 0] * kx4 / 8, uv[:, :, 1] * ky4 / 8).float()
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)  # 真实光流
        up_feat5 = self.upfeat5(x)

        warp4 = self.warp(c24, up_flow5 / 4)
        corr4 = self.corr(self.reduce4(c14), warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = self.conv4_0(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)

        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)  # 真实光流
        up_feat4 = self.upfeat4(x)

        warp3 = self.warp(c23, up_flow4 / 2)
        corr3 = self.corr(self.reduce3(c13), warp3)
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = self.conv3_0(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)  # 真实光流
        up_feat3 = self.upfeat3(x)

        warp2 = self.warp(c22, up_flow3)
        corr2 = self.corr(self.reduce2(c12), warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv2_4(x)
        flow2 = self.predict_flow2(x)
        # x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        # flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        # 真实光流
        weight = self.get_weight(x)

        kx4 = 256.0 / 1380.0
        ky4 = 64.0 / 384.0
        sparse_flow = bilinear_interpolate_torch(flow2, uv[:, :, 0] * kx4, uv[:, :, 1] * ky4).float()
        sparse_weight = bilinear_interpolate_torch(weight, uv[:, :, 0] * kx4, uv[:, :, 1] * ky4).float()
        # sparse_weight[:, :, :] = 1
        K4 = K.clone()
        K4[:, 0, 0] = K[:, 0, 0] * kx4
        K4[:, 0, 2] = K[:, 0, 2] * kx4
        K4[:, 1, 1] = K[:, 1, 1] * ky4
        K4[:, 1, 2] = K[:, 1, 2] * ky4
        T_dist, rigid_flow = self.newton_gauss(pcl_xyz, uv[:, :, 0] * kx4, uv[:, :, 1] * ky4, uv_t[:, :, 0] * kx4,
                                               uv_t[:, :, 1] * ky4, sparse_flow.clone(), sparse_weight, mask, K4)
        # x = x.view(x.shape[0], -1)  # 32,N
        # x = self.dropout(x)
        # x = self.leakyRELU(self.fc1(x))  # 32,N to 512
        #
        # transl = self.leakyRELU(self.fc1_trasl(x))  # 512 to 256 平移
        # rot = self.leakyRELU(self.fc1_rot(x))  # 512 to 256 旋转
        # transl = self.fc2_trasl(transl)  # 256 o 3
        # rot = self.fc2_rot(rot)  # 256 to 4
        # rot = F.normalize(rot, dim=1)  # 归一化
        sparse_flow[:, :, 0] = sparse_flow[:, :, 0] / kx4
        sparse_flow[:, :, 1] = sparse_flow[:, :, 1] / ky4

        sparse_flow3 = bilinear_interpolate_torch(up_flow3, uv[:, :, 0] * kx4, uv[:, :, 1] * ky4).float()
        sparse_flow4 = bilinear_interpolate_torch(up_flow4, uv[:, :, 0] * kx4 / 2, uv[:, :, 1] * ky4 / 2).float()
        sparse_flow5 = bilinear_interpolate_torch(up_flow5, uv[:, :, 0] * kx4 / 4, uv[:, :, 1] * ky4 / 4).float()
        sparse_flow6 = bilinear_interpolate_torch(up_flow6, uv[:, :, 0] * kx4 / 8, uv[:, :, 1] * ky4 / 8).float()
        sparse_flow6[:, :, 0] = sparse_flow6[:, :, 0] / kx4
        sparse_flow6[:, :, 1] = sparse_flow6[:, :, 1] / ky4

        sparse_flow5[:, :, 0] = sparse_flow5[:, :, 0] / kx4
        sparse_flow5[:, :, 1] = sparse_flow5[:, :, 1] / ky4

        sparse_flow4[:, :, 0] = sparse_flow4[:, :, 0] / kx4
        sparse_flow4[:, :, 1] = sparse_flow4[:, :, 1] / ky4

        sparse_flow3[:, :, 0] = sparse_flow3[:, :, 0] / kx4
        sparse_flow3[:, :, 1] = sparse_flow3[:, :, 1] / ky4
        return T_dist.inverse(), sparse_flow, sparse_flow3, sparse_flow4, sparse_flow5, sparse_flow6  # transl, rot
