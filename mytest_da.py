import os
import torch
import time
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib
import scipy.stats as st
import datasets
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

def align_shift_and_scale(gt_disp, pred_disp):

    t_gt = np.median(gt_disp)
    s_gt = np.mean(np.abs(gt_disp - t_gt))

    t_pred = np.median(pred_disp)
    s_pred = np.mean(np.abs(pred_disp - t_pred))
    print(t_gt, s_gt, t_pred, s_pred)
    pred_disp_aligned = (pred_disp - t_pred) * (s_gt / s_pred) + t_gt

    return pred_disp_aligned, t_gt, s_gt, t_pred, s_pred

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)
splits_dir = os.path.join(os.path.dirname(__file__), "splits")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# save_folder = 'pretrained_weight_da'
# save_path = os.path.join(save_folder, "depth_anything_{:}14.pth".format("vitl"))

# to_save = depth_anything.state_dict()
# for key in list(to_save.keys()):
#     if key.startswith('pretrained'):
#         new_key = key.replace('pretrained', 'encoder')
#         to_save[new_key] = to_save.pop(key)
# torch.save(to_save, save_path)

# image = torch.randn([4,3,224,280]).to(device)
# time_start = time.time()
# feature = depth_anything(image)
# inference_time = (time.time() - time_start)*1000

MIN_DEPTH = 1e-3
MAX_DEPTH = 150
eval_split = "endovis"
disable_median_scaling = False
save_vis = False
vis_dir = ("./vis")
os.makedirs(vis_dir, exist_ok=True)
test_dir = ("./test")
os.makedirs(test_dir, exist_ok=True)

filenames = readlines(os.path.join(splits_dir, "endovis", "test_files.txt"))
dataset = datasets.SCAREDRAWDataset('/mnt/data-hdd2/Beilei/Dataset/SCARED', filenames,
                                        256, 320,
                                        [0], 4, is_train=False)
dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=4,
                            pin_memory=True, drop_last=False)

## SCARED data
data_path = '/mnt/data-hdd2/Beilei/Dataset/SCARED'
filenames = readlines(os.path.join(splits_dir, "endovis", "test_files.txt"))

##

encoder = 'vitb' # can also be 'vits' or 'vitl'
depther = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder))
model_dict = depther.state_dict()
depther.cuda()
depther.eval()
gt_path = os.path.join(splits_dir, "endovis", "gt_depths.npz")
gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

inference_times = []
sequences = []
keyframes = []
frame_ids = []

errors = []
ratios = []
pred_disps = []
print("-> Computing predictions with size {}x{}".format(
    320, 256))

with torch.no_grad():
    for index in tqdm(range(len(filenames))):
        line = filenames[index].split()
        folder = line[0]
        sequence = folder[7]
        keyframe = folder[-1]
        str_sequence = str(np.array(int(sequence)))
        str_keyframe = str(np.array(int(keyframe)))
        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0
        
        str_frameid = "{:06d}".format(np.array(frame_index))
        f_str = "frame_data{:06d}{}".format(frame_index-1, '.png')
        sequence = folder[7]
        data_splt = "train" if int(sequence) < 8 else "test"
        image_path = os.path.join(
                data_path, data_splt, folder, "data", "left", f_str)
        raw_image = cv2.imread(image_path)
        image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        h, w = image.shape[:2]

        image = transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).cuda()
        time_start = time.time()
        output = depther(image)
        inference_time = time.time() - time_start
        output = torch.nn.functional.interpolate(output[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

        output_disp = output.cpu().numpy()
        # pred_disp, _ = disp_to_depth(output_disp, 0.1, 150)
        savesss = output_disp[np.newaxis, ...]
        pred_disps.append(savesss)
        pred_disp = savesss[0]
        print(pred_disp.shape)
        if save_vis:
            disp = (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min()) * 255.0
            disp = disp.astype(np.uint8)
            disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_INFERNO)
            vis_file_name = os.path.join(vis_dir, str_sequence + "_" +  str_keyframe + "_" + str_frameid + ".png")
            cv2.imwrite(vis_file_name, disp_color)
        
        inference_times.append(inference_time)


        gt_depth = gt_depths[index]
        gt_height, gt_width = gt_depth.shape[:2]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

        pred_disp = pred_disp[mask]
        gt_depth = gt_depth[mask]
        gt_disp = 1/gt_depth
        pred_disp_aligned, t_gt, s_gt, t_pred, s_pred = align_shift_and_scale(gt_disp, pred_disp)
        
        pred_depth_aligned = 1 / pred_disp_aligned
        pred_depth_aligned[pred_depth_aligned < MIN_DEPTH] = MIN_DEPTH
        pred_depth_aligned[pred_depth_aligned > MAX_DEPTH] = MAX_DEPTH
        error = compute_errors(gt_depth, pred_depth_aligned)
        if not np.isnan(error).all():
            errors.append(error)

pred_disps = np.concatenate(pred_disps)
output_path = "disps_endovis_split_DA.npy"
print("-> Saving predicted disparities to ", output_path)
np.save(output_path, pred_disps)

if not disable_median_scaling:
    ratios = np.array(ratios)
    med = np.median(ratios)
    print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

errors = np.array(errors)
mean_errors = np.mean(errors, axis=0)
cls = []
for i in range(len(mean_errors)):
    cl = st.t.interval(confidence=0.95, df=len(errors)-1, loc=mean_errors[i], scale=st.sem(errors[:,i]))
    cls.append(cl[0])
    cls.append(cl[1])
cls = np.array(cls)
print("\n       " + ("{:>11}      | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
print("mean:" + ("&{: 12.3f}      " * 7).format(*mean_errors.tolist()) + "\\\\")
print("cls: " + ("& [{: 6.3f}, {: 6.3f}] " * 7).format(*cls.tolist()) + "\\\\")
print("average inference time: {:0.1f} ms".format(np.mean(np.array(inference_times))*1000))
print("\n-> Done!")