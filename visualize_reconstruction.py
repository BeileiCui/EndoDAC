from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import open3d as o3d

from utils.layers import disp_to_depth
from utils.utils import readlines
from options import MonodepthOptions
import datasets
import models.encoders as encoders
import models.decoders as decoders
import models.endodac as endodac


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4

def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)

def compute_scale(gt, pred,min,max):
    mask = np.logical_and(gt > min, gt < max)
    pred = pred[mask]
    gt = gt[mask]
    scale = np.median(gt) / np.median(pred)
    return scale

def reconstruct_pointcloud(rgb, depth, cam_K, vis_rgbd=False):

    rgb = np.asarray(rgb, order="C")
    rgb_im = o3d.geometry.Image(rgb.astype(np.uint8))
    depth_im = o3d.geometry.Image(depth)
    
    print(rgb_im)
    print(depth_im)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_im, depth_im, convert_rgb_to_intensity=False)
    if vis_rgbd:
        plt.subplot(1, 2, 1)
        plt.title('RGB image')
        plt.imshow(rgbd_image.color)
        plt.subplot(1, 2, 2)
        plt.title('Depth image')
        plt.imshow(rgbd_image.depth)
        plt.colorbar()
        plt.show()

    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.intrinsic_matrix = cam_K
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        cam
    )

    return pcd
def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 150

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    if opt.model_type == 'endodac':
        depther_path = os.path.join(opt.load_weights_folder, "depth_model.pth")
        depther_dict = torch.load(depther_path)
    elif opt.model_type == 'afsfm':
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
        encoder_dict = torch.load(encoder_path)

    if opt.eval_split == 'endovis':
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "3d_reconstruction.txt"))
        dataset = datasets.SCAREDRAWDataset(opt.data_path, filenames,
                                        opt.height, opt.width,
                                        [0], 4, is_train=False)

    save_dir = os.path.join(opt.load_weights_folder, "reconstruction")
    os.makedirs(save_dir, exist_ok=True)
    
    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)
    if opt.model_type == 'endodac':
        depther = endodac.endodac(
            backbone_size = "base", r=opt.lora_rank, lora_type=opt.lora_type,
            image_shape=(224,280), pretrained_path=opt.pretrained_path,
            residual_block_indexes=opt.residual_block_indexes,
            include_cls_token=opt.include_cls_token)
        model_dict = depther.state_dict()
        depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
        depther.cuda()
        depther.eval()
    elif opt.model_type == 'afsfm':
        encoder = encoders.ResnetEncoder(opt.num_layers, False)
        depth_decoder = decoders.DepthDecoder(encoder.num_ch_enc, scales=range(4))
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
        depther = lambda image: depth_decoder(encoder(image))
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

    rgbs = []
    pred_disps = []
    cam_Ks = []
    inference_times = []
    sequences = []
    keyframes = []
    frame_ids = []

    print("-> Computing predictions with size {}x{}".format(
        opt.height, opt.width))

    with torch.no_grad():
        for data in tqdm(dataloader):
            input_color = data[("color", 0, 0)].cuda()
            if opt.post_process:
                # Post-processed results require each image to have two forward passes
                input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            time_start = time.time()
            output = depther(input_color)
            inference_time = time.time() - time_start
            pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.cpu()[:, 0].numpy()

            rgbs.append(input_color)
            pred_disps.append(pred_disp)
            cam_Ks.append(data[("K", 0)])
            inference_times.append(inference_time)
            sequences.append(data['sequence'])
            keyframes.append(data['keyframe'])
            frame_ids.append(data['frame_id'])

        
    pred_disps = np.concatenate(pred_disps)

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()
    elif opt.eval_split == 'endovis':
        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]


    if opt.visualize_depth:
        vis_dir = os.path.join(opt.load_weights_folder, "vis_depth")
        os.makedirs(vis_dir, exist_ok=True)
        
    print("-> Reconstructing")

    pcds = []
    for i in tqdm(range(pred_disps.shape[0])):

        sequence = str(np.array(sequences[i][0]))
        keyframe = str(np.array(keyframes[i][0]))
        frame_id = "{:06d}".format(frame_ids[i][0])
        
        pred_disp = pred_disps[i]
        pred_depth = 1/pred_disp
        pred_height, pred_width = pred_depth.shape[:2]

        gt_depth = gt_depths[i]
        gt_depth = cv2.resize(gt_depth, (pred_width, pred_height), interpolation=cv2.INTER_NEAREST)

        rgb = rgbs[i].squeeze().permute(1,2,0).cpu().numpy() * 255
        cam_K = cam_Ks[i][0,:3,:3].numpy()
        if opt.visualize_depth:
            vis_pred_depth = render_depth(pred_depth)
            vis_file_name = os.path.join(vis_dir, sequence + "_" +  keyframe + "_" + frame_id + ".png")
            vis_pred_depth.save(vis_file_name)

        scale = compute_scale(gt_depth, pred_depth, MIN_DEPTH ,MAX_DEPTH)
        pred_depth *= scale
        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        pcd = reconstruct_pointcloud(rgb, pred_depth, cam_K, vis_rgbd=False)
        o3d.visualization.draw_geometries([pcd])
        pcds.append(pcd)
        if i == 0:
            break
    print('Saving point clouds...')
    for i, pcd in enumerate(pcds):
        sequence = str(np.array(sequences[i][0]))
        keyframe = str(np.array(keyframes[i][0]))
        frame_id = "{:06d}".format(frame_ids[i][0])
        fn = os.path.join(save_dir, sequence + "_" +  keyframe + "_" + frame_id + ".ply")
        o3d.io.write_point_cloud(fn, pcd)
        
    
    print('Point clouds saved to', save_dir)


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
