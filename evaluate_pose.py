from __future__ import absolute_import, division, print_function

import os
import torch
import models.encoders as encoders
import models.decoders as decoders
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from utils.layers import transformation_from_parameters
from utils.utils import readlines
from options import MonodepthOptions
from datasets import SCAREDRAWDataset
import scipy.stats as st

# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        # cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


def dump_r(source_to_target_transformations):
    rs = []
    cam_to_world = np.eye(4)
    rs.append(cam_to_world[:3, :3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        # cam_to_world = np.dot(source_to_target_transformation, cam_to_world)
        rs.append(cam_to_world[:3, :3])
    return rs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):

    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def compute_re(gtruth_r, pred_r):
    RE = 0
    gt = gtruth_r
    pred = pred_r
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose @ np.linalg.inv(pred_pose)
        s = np.linalg.norm([R[0, 1] - R[1, 0],
                            R[1, 2] - R[2, 1],
                            R[0, 2] - R[2, 0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s, c)

    return RE / gtruth_r.shape[0]


def evaluate(opt):
    """Evaluate odometry on the SCARED dataset
    """
    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    filenames1 = readlines(
        os.path.join(os.path.dirname(__file__), "splits", "endovis",
                     "test_files_sequence1.txt"))
    filenames2 = readlines(
        os.path.join(os.path.dirname(__file__), "splits", "endovis",
                     "test_files_sequence2.txt"))
    
    dataset1 = SCAREDRAWDataset(opt.data_path, filenames1, opt.height, opt.width,
                               [0, 1], 4, is_train=False)
    dataloader1 = DataLoader(dataset1, 1, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    dataset2 = SCAREDRAWDataset(opt.data_path, filenames2, opt.height, opt.width,
                               [0, 1], 4, is_train=False)
    dataloader2 = DataLoader(dataset2, 1, shuffle=False,
                            num_workers=opt.num_workers, pin_memory=True, drop_last=False)
    
    pose_encoder_path = os.path.join(opt.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(opt.load_weights_folder, "pose.pth")
    intrinsics_decoder_path = os.path.join(opt.load_weights_folder, "intrinsics_head.pth")
    
    pose_encoder = encoders.ResnetEncoder(opt.num_layers, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    pose_decoder = decoders.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    if opt.learn_intrinsics:
        intrinsics_decoder = decoders.IntrinsicsHead(pose_encoder.num_ch_enc)
        intrinsics_decoder.load_state_dict(torch.load(intrinsics_decoder_path))
        intrinsics_decoder.cuda()
        intrinsics_decoder.eval()

    pose_encoder.cuda()
    pose_encoder.eval()
    pose_decoder.cuda()
    pose_decoder.eval()

    pred_poses_1 = []
    pred_intrinsics_1 = []
    pred_poses_2 = []
    pred_intrinsics_2 = []
    
    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in tqdm(dataloader1):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            all_color_aug = torch.cat([inputs[("color", 1, 0)], inputs[("color", 0, 0)]], 1)

            features = [pose_encoder(all_color_aug)]
            axisangle, translation, intermediate_feature = pose_decoder(features)
            
            pred_poses_1.append(
                transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
            
            if opt.learn_intrinsics:
                cam_K = intrinsics_decoder(
                        intermediate_feature, opt.width, opt.height)
                pred_intrinsics_1.append(cam_K[:,:3,:3].cpu().numpy())
                
        for inputs in tqdm(dataloader2):
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            all_color_aug = torch.cat([inputs[("color", 1, 0)], inputs[("color", 0, 0)]], 1)

            features = [pose_encoder(all_color_aug)]
            axisangle, translation, intermediate_feature = pose_decoder(features)
            
            pred_poses_2.append(
                transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())
            
            if opt.learn_intrinsics:
                cam_K = intrinsics_decoder(
                        intermediate_feature, opt.width, opt.height)
                pred_intrinsics_2.append(cam_K[:,:3,:3].cpu().numpy())
                
    pred_poses_1 = np.concatenate(pred_poses_1)
    pred_poses_2 = np.concatenate(pred_poses_2)
    if opt.learn_intrinsics:
        pred_intrinsics_1 = np.concatenate(pred_intrinsics_1)
        pred_intrinsics_2 = np.concatenate(pred_intrinsics_2)
        
    gt_path_1 = os.path.join(os.path.dirname(__file__), "splits", "endovis", "curve", "gt_poses_sequence1.npz")
    gt_local_poses_1 = np.load(gt_path_1, fix_imports=True, encoding='latin1')["data"]
    pred_path_1 = os.path.join(os.path.dirname(__file__), "splits", "endovis", "curve", "pred_poses_sequence1.npz")
    np.savez_compressed(pred_path_1, data=np.array(pred_poses_1))
    gt_path_2 = os.path.join(os.path.dirname(__file__), "splits", "endovis", "curve", "gt_poses_sequence2.npz")
    gt_local_poses_2 = np.load(gt_path_2, fix_imports=True, encoding='latin1')["data"]
    pred_path_2 = os.path.join(os.path.dirname(__file__), "splits", "endovis", "curve", "pred_poses_sequence2.npz")
    np.savez_compressed(pred_path_2, data=np.array(pred_poses_2))
    
    ates_1 = []
    res_1 = []
    ates_2 = []
    res_2 = []
    num_frames_1 = gt_local_poses_1.shape[0]
    num_frames_2 = gt_local_poses_2.shape[0]
    track_length = 5

    for i in range(0, num_frames_1 - 1):
        local_xyzs = np.array(dump_xyz(pred_poses_1[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses_1[i:i + track_length - 1]))
        local_rs = np.array(dump_r(pred_poses_1[i:i + track_length - 1]))
        gt_rs = np.array(dump_r(gt_local_poses_1[i:i + track_length - 1]))
        ates_1.append(compute_ate(gt_local_xyzs, local_xyzs))
        res_1.append(compute_re(local_rs, gt_rs))
    for i in range(0, num_frames_2 - 1):
        local_xyzs = np.array(dump_xyz(pred_poses_2[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses_2[i:i + track_length - 1]))
        local_rs = np.array(dump_r(pred_poses_2[i:i + track_length - 1]))
        gt_rs = np.array(dump_r(gt_local_poses_2[i:i + track_length - 1]))
        ates_2.append(compute_ate(gt_local_xyzs, local_xyzs))
        res_2.append(compute_re(local_rs, gt_rs))
        
    cls_1 = st.t.interval(alpha=0.95, df=len(ates_1)-1, loc=np.mean(ates_1), scale=st.sem(ates_1))
    cls_1 = np.array(cls_1)
    cls_2 = st.t.interval(alpha=0.95, df=len(ates_2)-1, loc=np.mean(ates_2), scale=st.sem(ates_2))
    cls_2 = np.array(cls_2)
    
    print("\n   sq1 Trajectory error: {:0.4f}, std: {:0.4f}, 95% cls: [{:0.4f}, {:0.4f}]\n".format(np.mean(ates_1), np.std(ates_1), cls_1[0], cls_1[1]))
    print("\n   sq1 Rotation error: {:0.4f}, std: {:0.4f}\n".format(np.mean(res_1), np.std(res_1)))
    print("\n   sq2 Trajectory error: {:0.4f}, std: {:0.4f}, 95% cls: [{:0.4f}, {:0.4f}]\n".format(np.mean(ates_2), np.std(ates_2), cls_2[0], cls_2[1]))
    print("\n   sq2 Rotation error: {:0.4f}, std: {:0.4f}\n".format(np.mean(res_2), np.std(res_2)))
    
    if opt.learn_intrinsics:
        pred_intrinsics = np.concatenate((pred_intrinsics_1, pred_intrinsics_2), axis=0)
        fx_mean, fx_std = np.mean(pred_intrinsics[:,0,0]) / opt.width, np.std(pred_intrinsics[:,0,0]) / opt.width
        fy_mean, fy_std = np.mean(pred_intrinsics[:,1,1]) / opt.height, np.std(pred_intrinsics[:,1,1]) / opt.height
        cx_mean, cx_std = np.mean(pred_intrinsics[:,0,2]) / opt.width, np.std(pred_intrinsics[:,0,2]) / opt.width
        cy_mean, cy_std = np.mean(pred_intrinsics[:,1,2]) / opt.height, np.std(pred_intrinsics[:,1,2]) / opt.height
        print("\n   fx: {:0.4f}, std: {:0.4f}\n".format(fx_mean, fx_std))
        print("\n   fy: {:0.4f}, std: {:0.4f}\n".format(fy_mean, fy_std))
        print("\n   cx: {:0.4f}, std: {:0.4f}\n".format(cx_mean, cx_std))
        print("\n   cy: {:0.4f}, std: {:0.4f}\n".format(cy_mean, cy_std))

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
