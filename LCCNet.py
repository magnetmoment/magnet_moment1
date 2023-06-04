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


class SpatialTransformer(nn.Module):
    """
    Implements a spatial transformer
    as proposed in the Jaderberg paper.
    Comprises of 3 parts:
    1. Localization Net
    2. A grid generator
    3. A roi pooled module.

    The current implementation uses a very small convolutional net with
    2 convolutional layers and 2 fully connected layers. Backends
    can be swapped in favor of VGG, ResNets etc. TTMV
    Returns:
    A roi feature map with the same input spatial dimension as the input feature map.
    """

    def __init__(self, in_channels, spatial_dims, kernel_size, use_dropout=False):
        super(SpatialTransformer, self).__init__()
        self._h, self._w = spatial_dims
        self._in_ch = in_channels
        self._ksize = kernel_size
        self.dropout = use_dropout

        # localization net
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=self._ksize, stride=1, padding=1,
                               bias=False)  # size : [1x3x32x32]
        self.conv2 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=self._ksize, stride=1, padding=1, bias=False)

        self.fc1 = nn.Linear(32 * 32 * 64, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):  # stn
        """
        Forward pass of the STN module.
        x -> input feature map
        """
        batch_images = x  # B，1，256，512
        x = F.relu(self.conv1(x.detach()))  # B，32，W，H
        x = F.relu(self.conv2(x))  # B,32,W,H
        x = F.max_pool2d(x, 2)  # B,32,W/2,H/2
        x = F.relu(self.conv3(x))  # B,32,W/2,H/2
        x = F.max_pool2d(x, 2)  # B,32,W/4,H/4
        x = F.relu(self.conv4(x))  # B,32,W/4,H/4
        x = F.max_pool2d(x, 2)  # B,32,W/8,H/8
        # x = F.relu(self.conv5(x))  # B,32,W/8,H/8
        # x = F.max_pool2d(x, 2)  # B,32,W/16,H/16
        # x = F.relu(self.conv6(x))  # B,32,W/16,H/16
        # x = F.max_pool2d(x, 2)  # B,32,W/32,H/32

        # print("Pre view size:{}".format(x.size()))  128 32 4 4
        x = x.view(-1, 32 * 32 * 64)  # B,N
        if self.dropout:
            x = F.dropout(self.fc1(x), p=0.5)
            x = F.dropout(self.fc2(x), p=0.5)
        else:
            x = self.fc1(x)  # B,1024
            x = self.fc2(x)  # params [Nx6] B,6

        x = x.view(-1, 1, 1)  # change it to the 2x3 matrix    hudu
        cs = torch.cos(x)
        ss = torch.sin(x)
        h = 1e-8 * torch.ones(x.shape[0], 2, x.shape[2]).to(x.device)
        x = torch.cat([torch.cat([cs, -ss], 1), torch.cat([ss, cs], 1)], 2)
        x = torch.cat([x, h], 2)
        # [[math.cos(degree), -math.sin(degree), 1e-8], [math.sin(degree), math.cos(degree), 1e-8]]

        affine_grid_points = F.affine_grid(x, torch.Size((x.size(0), self._in_ch, self._h, self._w)))  # 128，32，32，2
        assert (affine_grid_points.size(0) == batch_images.size(
            0)), "The batch sizes of the input images must be same as the generated grid."
        rois = F.grid_sample(batch_images, affine_grid_points)  # 128，3，32，32
        # print("rois found to be of size:{}".format(rois.size())) 128,3,32,32
        return rois


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

    x0_f = torch.clamp(x0, 0, im.shape[1] - 1).reshape(-1)
    x1_f = torch.clamp(x1, 0, im.shape[1] - 1).reshape(-1)
    y0_f = torch.clamp(y0, 0, im.shape[0] - 1).reshape(-1)
    y1_f = torch.clamp(y1, 0, im.shape[0] - 1).reshape(-1)

    B = torch.arange(0, batch).unsqueeze(-1).expand(-1, N).reshape(-1)
    Ia = im[B, :, y0_f, x0_f].reshape(batch, N, -1)
    Ib = im[B, :, y1_f, x0_f].reshape(batch, N, -1)
    Ic = im[B, :, y0_f, x1_f].reshape(batch, N, -1)
    Id = im[B, :, y1_f, x1_f].reshape(batch, N, -1)

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


# class rigid_flow(nn.Module):
#     def __init__(self, in_channels, original_img_size, cur_img_size, K):
#         super(rigid_flow, self).__init__()
#
#         # localization net
#         self.conv1_flow = nn.Conv1d(in_channels, 2, kernel_size=(1,), stride=(1,), bias=False)
#         self.conv1_weight = nn.Conv1d(in_channels, 1, kernel_size=(1,), stride=(1,), bias=False)
#
#     def get_rigid_flow(self, sparse_weight, K, current_shape, original_img_size):
#
#     def forward(self, feature, dense_flow, uv, pcl_xyz, mask):  # stn
#         # 1.取得点特征和点光流
#         sparse_flow = bilinear_interpolate_torch(dense_flow, uv[:, :, 0], uv[:, :, 1])
#         sparse_feature = bilinear_interpolate_torch(feature, uv[:, :, 0], uv[:, :, 1])
#         sparse_weight = self.conv1_weight(sparse_feature)
#         current_shape = feature.shape
#         # 2.用高斯牛顿法计算出 delta_T和刚性光流
#         delta_T, rigid_sparse_flow = get_rigid_flow(sparse_flow,\
#                             sparse_weight, K, current_shape, original_img_size)
#         # 3.融合光流
#
#
#         return


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

        # self.spatial_dim=image_size
        # self._in_ch=1
        # self._sksize=3
        #
        # # point stn
        # self.stnmod = SpatialTransformer(self._in_ch, self.spatial_dim, self._sksize)

        # resnet with leakyRELU
        self.Action_Func = Action_Func
        self.attention = attention
        self.inplanes = 64
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
        self.conv1_rgb = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.elu_rgb = nn.ELU()
        self.leakyRELU_rgb = nn.LeakyReLU(0.1)
        self.maxpool_rgb = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_rgb = self._make_layer(block, 64, layers[0])
        self.layer2_rgb = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_rgb = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_rgb = self._make_layer(block, 512, layers[3], stride=2)

        # lidar_image
        self.inplanes = 64
        self.conv1_lidar = nn.Conv2d(input_lidar, 64, kernel_size=7, stride=2, padding=3)
        self.elu_lidar = nn.ELU()
        self.leakyRELU_lidar = nn.LeakyReLU(0.1)
        self.maxpool_lidar = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_lidar = self._make_layer(block, 64, layers[0])
        self.layer2_lidar = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_lidar = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_lidar = self._make_layer(block, 512, layers[3], stride=2)

        self.corr = Correlation(pad_size=md, kernel_size=1, max_displacement=md, stride1=1, stride2=1,
                                corr_multiply=1)  # md=4
        self.leakyRELU = nn.LeakyReLU(0.1)

        nd = (2 * md + 1) ** 2
        dd = np.cumsum([128, 128, 96, 64, 32])

        od = nd
        self.conv6_0 = myconv(od, 128, kernel_size=3, stride=1)
        self.conv6_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv6_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv6_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv6_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)

        if use_feat_from > 1:
            self.predict_flow6 = predict_flow(od + dd[4])
            self.deconv6 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

            od = nd + add_list[0] + 4
            self.conv5_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv5_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
            self.conv5_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
            self.conv5_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
            self.conv5_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)

        if use_feat_from > 2:
            self.predict_flow5 = predict_flow(od + dd[4])
            self.deconv5 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

            od = nd + add_list[1] + 4
            self.conv4_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv4_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
            self.conv4_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
            self.conv4_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
            self.conv4_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)

        if use_feat_from > 3:
            self.predict_flow4 = predict_flow(od + dd[4])
            self.deconv4 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

            od = nd + add_list[2] + 4
            self.conv3_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv3_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
            self.conv3_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
            self.conv3_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
            self.conv3_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)

        if use_feat_from > 4:
            self.predict_flow3 = predict_flow(od + dd[4])
            self.deconv3 = deconv(2, 2, kernel_size=4, stride=2, padding=1)
            self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)

            od = nd + add_list[3] + 4
            self.conv2_0 = myconv(od, 128, kernel_size=3, stride=1)
            self.conv2_1 = myconv(od + dd[0], 128, kernel_size=3, stride=1)
            self.conv2_2 = myconv(od + dd[1], 96, kernel_size=3, stride=1)
            self.conv2_3 = myconv(od + dd[2], 64, kernel_size=3, stride=1)
            self.conv2_4 = myconv(od + dd[3], 32, kernel_size=3, stride=1)

        if use_feat_from > 5:
            self.predict_flow2 = predict_flow(od + dd[4])
            self.deconv2 = deconv(2, 2, kernel_size=4, stride=2, padding=1)

            self.dc_conv1 = myconv(od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1)
            self.dc_conv2 = myconv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
            self.dc_conv3 = myconv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
            self.dc_conv4 = myconv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
            self.dc_conv5 = myconv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
            self.dc_conv6 = myconv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
            self.dc_conv7 = predict_flow(32)
        self.get_weight = nn.Sequential(
            nn.Conv2d(od + dd[4], 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1, bias=True),
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

        self.fc1_trasl = nn.Linear(512, 256)
        self.fc1_rot = nn.Linear(512, 256)

        self.fc2_trasl = nn.Linear(256, 3)
        self.fc2_rot = nn.Linear(256, 4)

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
        vgrid = Variable(grid) + flo

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

    def newton_gauss(self, pc_f, flow, weight, mask, K):
        B = flow.shape[0]
        r = torch.zeros([B, 24000, 2, 1], dtype=torch.float32).cuda()  # B N 2 6
        J = torch.zeros([B, 24000, 2, 6], dtype=torch.float32).cuda()  # B N 2 6
        pc_f = torch.zeros([B, 4, 24000], dtype=torch.float32).cuda()  # B N 2 6
        pc_origin = torch.zeros([B, 4, 24000], dtype=torch.float32).cuda()  # B N 2 6
        u_gt = torch.zeros([B, 24000], dtype=torch.float32).cuda()  # B N 2 6
        v_gt = torch.zeros([B, 24000], dtype=torch.float32).cuda()
        fx = K[:, 0, 0]
        fy = K[:, 1, 1]

        for i in range(10):
            x = pc_f[:, :, 0]  # B N
            y = pc_f[:, :, 1]  # B N
            z = pc_f[:, :, 2]  # B N

            eps = 1e-7
            J[:, :, 0, 0] = fx / (z + eps)
            J[:, :, 0, 1] = 0
            J[:, :, 0, 2] = -fx * x / (z * z + eps)
            J[:, :, 0, 3] = fx * x * y / (z * z + eps)
            J[:, :, 0, 4] = fx + fx * x * x / (z * z + eps)
            J[:, :, 0, 5] = -fx * y / (z + eps)
            J[:, :, 1, 0] = 0
            J[:, :, 1, 1] = fy / (z + eps)
            J[:, :, 1, 2] = -fy * y / (z * z + eps)
            J[:, :, 1, 3] = -fy - fy * y * y / (z * z + eps)
            J[:, :, 1, 4] = fy * x * y / (z * z + eps)
            J[:, :, 1, 5] = fy * x / (z + eps)
            JT = J.permute(0, 1, 3, 2)
            proj = K.matmul(pc_f[:, 0:3, :])
            u = (proj[:, 0, :] / (proj[:, 2, :] + eps)).long()
            v = (proj[:, 1, :] / (proj[:, 2, :] + eps)).long()

            W = (num_mask * mask).float()  # 更新mask,这里还需要乘上网络更新的weight
            WH = W.unsqueeze(-1).unsqueeze(-1)
            Wb = W.unsqueeze(-1).unsqueeze(-1)

            H = JT.matmul(J)  # B N 6 6
            H = WH * H + 1e-05 * torch.eye(6).cuda()  # B N 6 6
            H = torch.mean(H, dim=1)
            H_1 = torch.cholesky_inverse(torch.linalg.cholesky(H))
            r[:, :, 0, 0] = u_gt - u  # B N 2 1
            r[:, :, 1, 0] = v_gt - v
            err = Wb * r

            b = Wb * JT.matmul(r)  # B N 6 1
            b = b.mean(dim=1)
            dx = H_1.matmul(b).squeeze(-1)  # 得到迭代结果 # B 6
            step = SE3.exp(dx).matrix()
            T_dist = SE3.exp(dx).matrix().matmul(T_dist)  # 更新 # B 4 4
            pc_f = T_dist.matmul(pc_origin)
            # delta_T = T_dist[0].inverse().matmul(RT)
            # print(step)
            # proj = proj.permute(0, 2, 1)
            delta_u = err[0, :, 0, 0]
            delta_v = err[0, :, 1, 0]
            print('t_step:{rot}, rot_step:{t}, err{err}'
                  .format(rot=dx[0, 0:3].norm(), t=dx[0, 3:6].norm(),
                          err=torch.sum(torch.abs(delta_u)) + torch.sum(torch.abs(delta_v))))

    def forward(self, rgb, lidar, uv, pcl_xyz, mask, K):  # B，N，H  插值
        # H, W = rgb.shape[2:4]
        # ig=rgb.cpu().numpy()[0].transpose(1,2,0)
        # # cv2.imwrite("{}/img.png".format(savepath),ig)
        # li=lidar.cpu().numpy()[0].transpose(1,2,0)
        # # cv2.imwrite("{}/lidar.png".format(savepath),li)
        # plt.imshow(ig)
        # plt.axis("off")
        # plt.savefig("{}/img.png".format(savepath))
        # # plt.show()
        # plt.imshow(li)
        # plt.axis("off")
        # plt.savefig("{}/lidar.png".format(savepath))
        # plt.show()
        # cv2.imshow('lidar',ig)
        # cv2.imshow('lidar2', li)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # encoder
        if self.pretrained_encoder:  # TRUE
            # rgb_image  1，3，256，512
            features1 = self.net_encoder(rgb)  # snet encode  6 feature 64，64，64，128，256，512
            c12 = features1[0]  # 2 1，64，128，256
            # draw_features(8, 8, c12.cpu().numpy(), "{}/img_c12.png".format(savepath))
            c13 = features1[2]  # 4 1，64，64，128
            # draw_features(8, 8, c13.cpu().numpy(), "{}/img_c13.png".format(savepath))
            c14 = features1[3]  # 8 1,128，32，64
            # draw_features(8, 16, c14.cpu().numpy(), "{}/img_c14.png".format(savepath))
            c15 = features1[4]  # 16 1，256，16，32
            # draw_features(16, 16, c15.cpu().numpy(), "{}/img_c15.png".format(savepath))
            c16 = features1[5]  # 32 1，512，8，16
            # draw_features(16, 32, c16.cpu().numpy(), "{}/img_c16.png".format(savepath))

            # rot_lidar = self.stnmod(lidar)

            # lidar_image
            x2 = self.conv1_lidar(lidar)  # 1，64，128，256
            # draw_features(8, 8, x2.cpu().numpy(), "{}/lidar_x2.png".format(savepath))
            if self.Action_Func == 'leakyrelu':
                c22 = self.leakyRELU_lidar(x2)  # 2
            elif self.Action_Func == 'elu':
                c22 = self.elu_lidar(x2)  # 2 1，64，128，256
            # draw_features(8, 8, c22.cpu().numpy(), "{}/lidar_c22.png".format(savepath))
            c23 = self.layer1_lidar(self.maxpool_lidar(c22))  # 4 pool 32,64,64,128
            # draw_features(8, 8, c23.cpu().numpy(), "{}/lidar_c23.png".format(savepath))
            c24 = self.layer2_lidar(c23)  # 8  32，128，32，64
            # draw_features(8, 16, c24.cpu().numpy(), "{}/lidar_c24.png".format(savepath))
            c25 = self.layer3_lidar(c24)  # 16  32，256，16，32
            # draw_features(16, 16, c25.cpu().numpy(), "{}/lidar_c25.png".format(savepath))
            c26 = self.layer4_lidar(c25)  # 32  32，512，8，16
            # draw_features(16, 32, c26.cpu().numpy(), "{}/lidar_c26.png".format(savepath))

        else:
            x1 = self.conv1_rgb(rgb)
            x2 = self.conv1_lidar(lidar)
            if self.Action_Func == 'leakyrelu':
                c12 = self.leakyRELU_rgb(x1)  # 2
                c22 = self.leakyRELU_lidar(x2)  # 2
            elif self.Action_Func == 'elu':
                c12 = self.elu_rgb(x1)  # 2
                c22 = self.elu_lidar(x2)  # 2
            c13 = self.layer1_rgb(self.maxpool_rgb(c12))  # 4
            c23 = self.layer1_lidar(self.maxpool_lidar(c22))  # 4
            c14 = self.layer2_rgb(c13)  # 8
            c24 = self.layer2_lidar(c23)  # 8
            c15 = self.layer3_rgb(c14)  # 16
            c25 = self.layer3_lidar(c24)  # 16
            c16 = self.layer4_rgb(c15)  # 32
            c26 = self.layer4_lidar(c25)  # 32

        corr6 = self.corr(c16, c26)  # 最后一层  1，512，8，16  O   1，81，8，16
        # draw_features(9, 9, c23.cpu().numpy(), "{}/corr6.png".format(savepath))
        corr6 = self.leakyRELU(corr6)
        x = torch.cat((self.conv6_0(corr6), corr6), 1)  # 32，81+128,8,16   209
        # draw_features(16, 16, x.cpu().numpy(), "{}/x_1.png".format(savepath))
        x = torch.cat((self.conv6_1(x), x), 1)  # 32,81+128+128,8,16      337
        # draw_features(20, 20, x.cpu().numpy(), "{}/x_2.png".format(savepath))
        x = torch.cat((self.conv6_2(x), x), 1)  # 32,81+128+128+96,8,16   433
        # draw_features(25, 25, x.cpu().numpy(), "{}/x_3.png".format(savepath))
        x = torch.cat((self.conv6_3(x), x), 1)  # 32,81+128+128+96+64,8,16 497
        # draw_features(25, 25, x.cpu().numpy(), "{}/x_4.png".format(savepath))
        x = torch.cat((self.conv6_4(x), x), 1)  # 32,81+128+128+96+64+32,8,16 529
        # draw_features(25, 25, x.cpu().numpy(), "{}/x_5.png".format(savepath))
        # use_feat_from=1
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)  # 我们直接回归出点云在当前尺寸图像下的真实光流
        up_feat6 = self.upfeat6(x)

        warp5 = self.warp(c25, up_flow6)
        corr5 = self.corr(c15, warp5)
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)

        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)  # 真实光流
        up_feat5 = self.upfeat5(x)

        warp4 = self.warp(c24, up_flow5)
        corr4 = self.corr(c14, warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)

        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)  # 真实光流
        up_feat4 = self.upfeat4(x)

        warp3 = self.warp(c23, up_flow4)
        corr3 = self.corr(c13, warp3)
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)

        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)  # 真实光流
        up_feat3 = self.upfeat3(x)

        warp2 = self.warp(c22, up_flow3)
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)
        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 = flow2 + self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        # 真实光流
        weight = self.get_weight(x)

        sparse_flow = bilinear_interpolate_torch(flow2, uv[:, :, 0] * 256.0 / 1380.0,
                                                 uv[:, :, 1] * 128.0 / 384.0)
        sparse_weight = bilinear_interpolate_torch(weight, uv[:, :, 0] * 256.0 / 1380.0,
                                                   uv[:, :, 1] * 128.0 / 384.0)
        self.newton_gauss(pcl_xyz, sparse_flow, sparse_weight, mask, K)
        # x = x.view(x.shape[0], -1)  # 32,N
        # x = self.dropout(x)
        # x = self.leakyRELU(self.fc1(x))  # 32,N to 512
        #
        # transl = self.leakyRELU(self.fc1_trasl(x))  # 512 to 256 平移
        # rot = self.leakyRELU(self.fc1_rot(x))  # 512 to 256 旋转
        # transl = self.fc2_trasl(transl)  # 256 o 3
        # rot = self.fc2_rot(rot)  # 256 to 4
        # rot = F.normalize(rot, dim=1)  # 归一化

        return flow5, flow4, flow3, flow2, sparse_flow  # transl, rot
