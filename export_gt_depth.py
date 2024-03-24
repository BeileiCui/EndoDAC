from __future__ import absolute_import, division, print_function

import os

import argparse
import numpy as np
import PIL.Image as pil
import cv2

from utils.utils import readlines
# from kitti_utils import generate_depth_map


def export_gt_depths_kitti():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["eigen", "eigen_benchmark", "endovis"])
    parser.add_argument('--useage',
                        type=str,
                        help='gt depth use for evaluation or 3d reconstruction',
                        required=True,
                        choices=["eval", "3d_recon"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    if opt.useage == "eval":
        lines = readlines(os.path.join(split_folder, "test_files.txt"))
        output_path = os.path.join(split_folder, "gt_depths.npz")
    else:
        lines = readlines(os.path.join(split_folder, "3d_reconstruction.txt"))
        output_path = os.path.join(split_folder, "gt_depths_recon.npz")
        
    print("Exporting ground truth depths for {}".format(opt.split))
    i=0
    gt_depths = []
    for line in lines:
        i = i+1
        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)
        print(i)
        print(folder)

        if opt.split == "eigen":
            calib_dir = os.path.join(opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            # gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(opt.data_path, folder, "proj_depth",
                                         "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256
        elif opt.split == "endovis":
            f_str = "scene_points{:06d}.tiff".format(frame_id-1)
            sequence = folder[7]
            data_splt = "train" if int(sequence) < 8 else "test"
            # gt_depth_path = os.path.join(
            # opt.data_path, data_splt, folder, "data", "left_depth",
            # f_str)
            # gt_depth = cv2.imread(gt_depth_path, 2)
            
            gt_depth_path = os.path.join(
            opt.data_path, data_splt, folder, "data", "scene_points",
            f_str)

            gt_depth = cv2.imread(gt_depth_path, 3)
            gt_depth = gt_depth[:, :, 0]
            gt_depth = gt_depth[0:1024, :]

        gt_depths.append(gt_depth.astype(np.float32))

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_kitti()
