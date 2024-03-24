from __future__ import absolute_import, division, print_function

import os,sys
import json
import argparse

from utils import readlines
from utils.layers import *


def export_gt_depths_SCARED():

    parser = argparse.ArgumentParser(description='export_gt_pose')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the KITTI data',
                        default='/mnt/data-hdd2/Beilei/Dataset/SCARED')
    parser.add_argument('--sequence',
                        type=str,
                        help='export gt from which sequence',
                        choices=["sequence1", "sequence2"],
                        default="sequence2")
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        choices=["eigen", "eigen_benchmark", "endovis"],
                        default="endovis")
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files_"+opt.sequence+ ".txt"))
    print("Exporting ground truth depths for {}".format(opt.split))

    gt_Ts = []
    for line in lines:
        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

        # pose
        f_str_0 = "frame_data{:06d}.json".format(frame_id - 1)
        f_str_1 = "frame_data{:06d}.json".format(frame_id)
        sequence = folder[7]
        data_splt = "train" if int(sequence) < 8 else "test"
        path_0 = os.path.join(
            opt.data_path, data_splt, folder, "data", "frame_data", f_str_0)
        path_1 = os.path.join(
            opt.data_path, data_splt, folder, "data", "frame_data", f_str_1)
        with open(path_0, 'r') as path0:
            data_0 = json.load(path0)
        with open(path_1,'r') as path1:
            data_1 = json.load(path1)
        data_0 = np.linalg.pinv(np.array(data_0['camera-pose']))
        data_1 = np.array(data_1['camera-pose'])
        T = (data_1 @ data_0).astype(np.float32)

        gt_Ts.append(T)

    output_path = os.path.join(split_folder, "curve", "gt_poses_"+opt.sequence+".npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_Ts))
    print("Done!")


if __name__ == "__main__":
    export_gt_depths_SCARED()
