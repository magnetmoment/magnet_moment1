# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/DatasetVisibilityKitti.py

import csv
from http.client import PRECONDITION_REQUIRED
import os
from math import radians
import cv2

import h5py
import mathutils
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
import time

from utils import invert_pose, rotate_forward, quaternion_from_matrix, rotate_back
from pykitti import odometry
import pykitti

torch.set_num_threads(1)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def get_2D_lidar_projection(pcl, cam_intrinsic):
    pcl_xyz = cam_intrinsic @ pcl.T
    pcl_xyz = pcl_xyz.T
    pcl_z = pcl_xyz[:, 2]
    pcl_xyz = pcl_xyz / (pcl_xyz[:, 2, None] + 1e-10)
    pcl_uv = pcl_xyz[:, :2]

    return pcl_uv, pcl_z


def lidar_project_depth(pc_rotated, cam_calib, img_shape):
    pc_rotated = pc_rotated[:3, :].detach().cpu().numpy()
    cam_intrinsic = cam_calib
    pcl_uv, pcl_z = get_2D_lidar_projection(pc_rotated.T, cam_intrinsic)
    mask = (pcl_uv[:, 0] > 0) & (pcl_uv[:, 0] < img_shape[1]) & (pcl_uv[:, 1] > 0) & (
            pcl_uv[:, 1] < img_shape[0]) & (pcl_z > 0)
    pcl_uv = pcl_uv[mask]
    pcl_z = pcl_z[mask]
    pcl_xy = pc_rotated[0:2, mask].transpose(1,0)
    pcl_xyz = np.concatenate([pcl_xy, pcl_z[:,None]], axis = 1)
    pcl_uv = pcl_uv.astype(np.uint32)
    pcl_z = pcl_z.reshape(-1, 1)
    depth_img = np.zeros((img_shape[0], img_shape[1], 1))
    depth_img[pcl_uv[:, 1], pcl_uv[:, 0]] = pcl_z
    depth_img = torch.from_numpy(depth_img.astype(np.float32))
    depth_img = depth_img
    depth_img = depth_img.permute(2, 0, 1)

    return depth_img, pcl_uv, pcl_xyz, mask


class DatasetLidarCameraKittiOdometry(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=20., split='val', device='cpu', val_sequence='00', suf='.png', val=False,
                 dataset=None, config=None, img_shape=None, max_points=20000, input_size=None):
        super(DatasetLidarCameraKittiOdometry, self).__init__()
        self._config = config
        self.img_shape = img_shape
        self.max_points = max_points
        self.input_size = input_size
        print('excute kitti dataset init')
        self.__init_kitti_data__(dataset_dir, transform, augmentation, use_reflectance,
                                 max_t, max_r, split, device, val_sequence, suf)

    def __init_kitti_data__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                            max_t=1.5, max_r=20., split='val', device='cpu', val_sequence='00', suf='.png'):
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.K = {}
        self.suf = suf

        self.all_files = []
        self.sequence_list = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                              '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
        # self.model = CameraModel()
        # self.model.focal_length = [7.18856e+02, 7.18856e+02]
        # self.model.principal_point = [6.071928e+02, 1.852157e+02]
        # for seq in ['00', '03', '05', '06', '07', '08', '09']:
        for seq in self.sequence_list:
            odom = odometry(self.root_dir, seq)
            calib = odom.calib
            T_cam02_velo_np = calib.T_cam2_velo  # gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)
            self.K[seq] = calib.K_cam2  # 3x3
            # T_cam02_velo = torch.from_numpy(T_cam02_velo_np)
            # GT_R = quaternion_from_matrix(T_cam02_velo[:3, :3])
            # GT_T = T_cam02_velo[3:, :3]
            # self.GTs_R[seq] = GT_R # GT_R = np.array([row['qw'], row['qx'], row['qy'], row['qz']])
            # self.GTs_T[seq] = GT_T # GT_T = np.array([row['x'], row['y'], row['z']])
            self.GTs_T_cam02_velo[seq] = T_cam02_velo_np  # gt pose from cam02 to velo_lidar (T_cam02_velo: 4x4)

            image_list = os.listdir(os.path.join(dataset_dir, 'sequences', seq, 'image_2'))
            image_list.sort()

            for image_name in image_list:

                if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'velodyne',
                                                   str(image_name.split('.')[0]) + '.bin')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'image_2',
                                                   str(image_name.split('.')[0]) + suf)):
                    continue
                if seq == val_sequence:
                    if split.startswith('val') or split == 'test':
                        self.all_files.append(os.path.join(seq, image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train':
                    self.all_files.append(os.path.join(seq, image_name.split('.')[0]))

        self.val_RT = []
        if split == 'val' or split == 'test':
            # val_RT_file = os.path.join(dataset_dir, 'sequences',
            #                            f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            val_RT_file = os.path.join(dataset_dir, 'sequences',
                                       f'val_RT_left_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'VAL SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                          rotx, roty, rotz])
                    self.val_RT.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                        float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        # rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            # io.imshow(np.array(rgb))
            # io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        item = self.all_files[idx]
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])
        img_path = os.path.join(self.root_dir, 'sequences', seq, 'image_2', rgb_name + self.suf)
        lidar_path = os.path.join(self.root_dir, 'sequences', seq, 'velodyne', rgb_name + '.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        if pc.shape[0] >= self.max_points:
            pc = pc[:self.max_points, :]
        else:
            pc = np.pad(pc, [[0, self.max_points - pc.shape[0]], [0, 0]], constant_values=0)
        pc_org = torch.from_numpy(pc.astype(np.float32))
        # if self.use_reflectance:
        #     reflectance = pc[:, 3].copy()
        #     reflectance = torch.from_numpy(reflectance).float()

        RT = self.GTs_T_cam02_velo[seq].astype(np.float32)

        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")
        pc_rot = np.matmul(RT, pc_org.numpy())
        # a = (RT @ pc_org.T).T
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        # pc_rot = np.matmul(RT, pc.T)
        # pc_rot = pc_rot.astype(np.float).T.copy()
        # pc_in = torch.from_numpy(pc_rot.astype(np.float32))#.float()

        # if pc_in.shape[1] == 4 or pc_in.shape[1] == 3:
        #     pc_in = pc_in.t()
        # if pc_in.shape[0] == 3:
        #     homogeneous = torch.ones(pc_in.shape[1]).unsqueeze(0)
        #     pc_in = torch.cat((pc_in, homogeneous), 0)
        # elif pc_in.shape[0] == 4:
        #      if not torch.all(pc_in[3,:] == 1.):
        #         pc_in[3,:] = 1.
        # else:
        #     raise TypeError("Wrong PointCloud shape")

        h_mirror = False
        # if np.random.rand() > 0.5 and self.split == 'train':
        #     h_mirror = True
        #     pc_in[1, :] *= -1

        img = Image.open(img_path)
        # img = cv2.imread(img_path)
        img_rotation = 0.
        # if self.split == 'train':
        #     img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        if self.split == 'train':
            R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        # io.imshow(depth_img.numpy(), cmap='jet')
        # io.show()
        calib = self.K[seq]
        if h_mirror:
            calib[2] = (img.shape[2] / 2) * 2 - calib[2]

        real_shape = [img.shape[1], img.shape[2], img.shape[0]]

        # pc_in = pc_in.cuda() # 变换到相机坐标系下的激光雷达点云
        pc_lidar = pc_in.clone()

        if self._config['max_depth'] < 80.:
            pc_lidar = pc_lidar[:, pc_lidar[0, :] < self._config['max_depth']].clone()

        depth_gt, uv, pcl_xyz = lidar_project_depth(pc_lidar, calib, real_shape)  # image_shape
        depth_gt /= self._config['max_depth']

        RR = mathutils.Quaternion(R).to_matrix()
        RR.resize_4x4()
        TT = mathutils.Matrix.Translation(T)
        RT_ = TT * RR

        pc_rotated = rotate_back(pc_in, RT_)  # Pc` = RT * Pc

        if self._config['max_depth'] < 80.:
            pc_rotated = pc_rotated[:, pc_rotated[0, :] < self._config['max_depth']].clone()

        depth_img, uv, pcl_xyz, mask = lidar_project_depth(pc_rotated, calib, real_shape)  # image_shape
        depth_img /= self._config['max_depth']

        # PAD ONLY ON RIGHT AND BOTTOM SIDE 这样不会改变内参
        rgb = img
        shape_pad = [0, 0, 0, 0]

        shape_pad[3] = (self.img_shape[0] - rgb.shape[1])  # // 2
        shape_pad[1] = (self.img_shape[1] - rgb.shape[2])  # // 2 + 1

        # print(f'shape_padL{shape_pad}')
        rgb = F.pad(rgb, shape_pad)
        depth_img = F.pad(depth_img, shape_pad)
        depth_gt = F.pad(depth_gt, shape_pad)

        uv2 = np.zeros((self.max_points, 2), dtype = np.int64)
        pcl_xyz2 = np.zeros((self.max_points, 3), dtype=np.int64)
        uv2[0 : uv.shape[0], :] = uv
        pcl_xyz2[0 : uv.shape[0], :] = pcl_xyz
        #16 32, 32 64, 64 128, 128 256,
        if self.split == 'test':
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'seq': int(seq), 'img_path': img_path,
                      'rgb_name': rgb_name + '.png', 'item': item, 'extrin': RT,
                      'initial_RT': initial_RT, 'uv': uv2 'mask': mask,'pcl_xyz' : pcl_xyz2}
        else:
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'seq': int(seq),
                      'rgb_name': rgb_name, 'item': item, 'extrin': RT, 'lidar_input': depth_img,
                      'rgb_input': rgb, 'pc_rotated': pc_rotated, 'shape_pad': shape_pad,
                      'real_shape': real_shape, 'depth_gt': depth_gt, 'uv': uv2,
                      'mask':mask, 'pcl_xyz':pcl_xyz2}

        return sample


class DatasetLidarCameraKittiRaw(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=15.0, split='val', device='cpu', val_sequence='2011_09_26_drive_0117_sync'):
        super(DatasetLidarCameraKittiRaw, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        self.GTs_R = {}
        self.GTs_T = {}
        self.GTs_T_cam02_velo = {}
        self.max_depth = 80
        self.K_list = {}

        self.all_files = []
        date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']
        data_drive_list = ['0001', '0002', '0004', '0016', '0027']
        self.calib_date = {}

        for i in range(len(date_list)):
            date = date_list[i]
            data_drive = data_drive_list[i]
            data = pykitti.raw(self.root_dir, date, data_drive)
            calib = {'K2': data.calib.K_cam2, 'K3': data.calib.K_cam3,
                     'RT2': data.calib.T_cam2_velo, 'RT3': data.calib.T_cam3_velo}
            self.calib_date[date] = calib

        # date = val_sequence[:10]
        # seq = val_sequence
        # image_list = os.listdir(os.path.join(dataset_dir, date, seq, 'image_02/data'))
        # image_list.sort()
        #
        # for image_name in image_list:
        #     if not os.path.exists(os.path.join(dataset_dir, date, seq, 'velodyne_points/data',
        #                                        str(image_name.split('.')[0]) + '.bin')):
        #         continue
        #     if not os.path.exists(os.path.join(dataset_dir, date, seq, 'image_02/data',
        #                                        str(image_name.split('.')[0]) + '.jpg')):  # png
        #         continue
        #     self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))

        date = val_sequence[:10]
        test_list = ['2011_09_26_drive_0005_sync', '2011_09_26_drive_0070_sync', '2011_10_03_drive_0027_sync']
        seq_list = os.listdir(os.path.join(self.root_dir, date))

        for seq in seq_list:
            if not os.path.isdir(os.path.join(dataset_dir, date, seq)):
                continue
            image_list = os.listdir(os.path.join(dataset_dir, date, seq, 'image_02/data'))
            image_list.sort()

            for image_name in image_list:
                if not os.path.exists(os.path.join(dataset_dir, date, seq, 'velodyne_points/data',
                                                   str(image_name.split('.')[0]) + '.bin')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, date, seq, 'image_02/data',
                                                   str(image_name.split('.')[0]) + '.jpg')):  # png
                    continue
                if seq == val_sequence and (not split == 'train'):
                    self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train' and seq not in test_list:
                    self.all_files.append(os.path.join(date, seq, 'image_02/data', image_name.split('.')[0]))

        self.val_RT = []
        if split == 'val' or split == 'test':
            val_RT_file = os.path.join(dataset_dir,
                                       f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            if os.path.exists(val_RT_file):
                print(f'VAL SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'TEST SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
                for i in range(len(self.all_files)):
                    rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                    transl_x = np.random.uniform(-max_t, max_t)
                    transl_y = np.random.uniform(-max_t, max_t)
                    transl_z = np.random.uniform(-max_t, max_t)
                    # transl_z = np.random.uniform(-max_t, min(max_t, 1.))
                    val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                          rotx, roty, rotz])
                    self.val_RT.append([float(i), transl_x, transl_y, transl_z,
                                        rotx, roty, rotz])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"

    def get_ground_truth_poses(self, sequence, frame):
        return self.GTs_T[sequence][frame], self.GTs_R[sequence][frame]

    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        # rgb = crop(rgb )
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            # color_transform = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            # io.imshow(np.array(rgb))
            # io.show()

        rgb = to_tensor(rgb)
        # rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    # self.all_files.append(os.path.join(date, seq, 'image_2/data', image_name.split('.')[0]))
    def __getitem__(self, idx):
        item = self.all_files[idx]
        date = str(item.split('/')[0])
        seq = str(item.split('/')[1])
        rgb_name = str(item.split('/')[4])
        img_path = os.path.join(self.root_dir, date, seq, 'image_02/data', rgb_name + '.jpg')  # png
        lidar_path = os.path.join(self.root_dir, date, seq, 'velodyne_points/data', rgb_name + '.bin')
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_lidar = pc.copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))
        if self.use_reflectance:
            reflectance = pc[:, 3].copy()
            reflectance = torch.from_numpy(reflectance).float()

        calib = self.calib_date[date]
        RT_cam02 = calib['RT2'].astype(np.float32)
        # camera intrinsic parameter
        calib_cam02 = calib['K2']  # 3x3

        E_RT = RT_cam02

        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")

        pc_rot = np.matmul(E_RT, pc_org.numpy())
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        h_mirror = False
        # if np.random.rand() > 0.5 and self.split == 'train':
        #     h_mirror = True
        #     pc_in[0, :] *= -1

        img = Image.open(img_path)
        # print(img_path)
        # img = cv2.imread(img_path)
        img_rotation = 0.
        # if self.split == 'train':
        #     img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        # if self.split == 'train':
        #     R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
        #     T = mathutils.Vector((0., 0., 0.))
        #     pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
            initial_RT = 0
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # 随机设置一定范围内的标定参数扰动值
        # train的时候每次都随机生成,每个epoch使用不同的参数
        # test则在初始化的时候提前设置好,每个epoch都使用相同的参数
        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        # io.imshow(depth_img.numpy(), cmap='jet')
        # io.show()
        calib = calib_cam02
        # calib = get_calib_kitti_odom(int(seq))
        if h_mirror:
            calib[2] = (img.shape[2] / 2) * 2 - calib[2]

        # sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
        #           'tr_error': T, 'rot_error': R, 'seq': int(seq), 'rgb_name': rgb_name, 'item': item,
        #           'extrin': E_RT, 'initial_RT': initial_RT}
        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'pc_org': pc_org, 'img_path': img_path,
                  'tr_error': T, 'rot_error': R, 'rgb_name': rgb_name + '.png', 'item': item,
                  'extrin': E_RT, 'initial_RT': initial_RT, 'pc_lidar': pc_lidar}

        return sample


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
