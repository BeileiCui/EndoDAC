from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import cv2
import json

from .mono_dataset import MonoDataset


class SCAREDDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.82, 0, 0.5, 0],
                           [0, 1.02, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        # self.full_res_shape = (1280, 1024)
        self.side_map = {"l": "left", "r": "right"}
    def check_depth(self):
        
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))
        
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class SCAREDRAWDataset(SCAREDDataset):
    def __init__(self, *args, **kwargs):
        super(SCAREDRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "frame_data{:06d}{}".format(frame_index-1, self.img_ext)
        sequence = folder[7]
        data_splt = "train" if int(sequence) < 8 else "test"
        image_path = os.path.join(
            self.data_path, data_splt, folder, "data", self.side_map[side], f_str)

        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "scene_points{:06d}.tiff".format(frame_index-1)
        sequence = folder[7]
        data_splt = "train" if int(sequence) < 8 else "test"
        # depth_path = os.path.join(
        #     self.data_path, data_splt, folder, "data", self.side_map[side] + "_depth",
        #     f_str)

        # depth_gt = cv2.imread(depth_path, 2)
        
        depth_path = os.path.join(
            self.data_path, data_splt, folder, "data", "scene_points",
            f_str)

        depth_gt = cv2.imread(depth_path, 3)
        depth_gt = depth_gt[:, :, 0]
        depth_gt = depth_gt[0:1024, :]
        
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
    
    def get_pose(self, folder, frame_index):
        f_str = "frame_data{:06d}.json".format(frame_index-1)
        sequence = folder[7]
        data_splt = "train" if int(sequence) < 8 else "test"
        pose_path = os.path.join(
            self.data_path, data_splt, folder, "data", "frame_data",
            f_str)
        with open(pose_path, 'r') as path:
            data = json.load(path)
            pose = np.linalg.pinv(np.array(data['camera-pose']))
        
        return pose


