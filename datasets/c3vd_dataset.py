from __future__ import absolute_import, division, print_function

import glob
import os
import random
import numpy as np
from PIL import Image  # using pillow-simd for increased speed
from PIL import ImageFile
import cv2

import torch.utils.data as data
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES=True

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
class C3VDDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False):
        super(C3VDDataset, self).__init__()

        self.data_path = data_path
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.LANCZOS

        self.frame_idxs = frame_idxs

        self.is_train = is_train

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.rescale_factor = 100 / 65535
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.transforms.ColorJitter(self.brightness,self.contrast,self.saturation,self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
        self.scans = []
        self.video_files = [os.path.join(self.data_path, file) for file in os.listdir(self.data_path)]
        self.video_files.sort()
        self.sequence_len = np.zeros([len(self.video_files)])
        for i, video_file in enumerate(self.video_files):
            image_paths = os.path.join(video_file, "*_color.png")
            video_base = os.path.basename(video_file)
            seq_image_paths = glob.glob(image_paths)
            seq_image_paths.sort()
            for seq_image_path in seq_image_paths:
                filename = os.path.basename(seq_image_path)
                seq_depth_path = os.path.join(video_file, filename[:-10]+"_depth.tiff")
                seq_normal_path = os.path.join(video_file, filename[:-10]+"_normals.tiff")
                seq_occlusion_path = os.path.join(video_file, filename[:-10]+"_occlusion.png")
                if os.path.exists(seq_image_path) and os.path.exists(seq_depth_path):
                    self.scans.append(
                        {
                            "image": seq_image_path,
                            "depth": seq_depth_path,
                            "normal": seq_normal_path,
                            "occlusion": seq_occlusion_path,
                            "sequence": video_base,
                            "index":filename[:-10],
                            "length": len(seq_image_paths),
                        })
        self.box = (200, 180, 1150, 900)
        print("Prepared C3VD dataset with %d sets of RGB, depth, normal and occlusion images." % (len(self.scans)))

    def __len__(self):
        return len(self.scans)
    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
    def get_color(self, path, do_flip):
        color = self.loader(path)
        
        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, path, do_flip):

        depth_gt = cv2.imread(path, 3)
        depth_gt = np.array(depth_gt[:, :, 0])
        depth_gt = depth_gt.astype(np.float32, order='C') * self.rescale_factor
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        scan = self.scans[index]
        image_dir = scan["image"]
        depth_dir = scan["depth"]
        inputs = {}
        
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        inputs[("color", 0, 0)] = self.get_color(image_dir, do_flip)
        inputs["depth_gt"] = self.get_depth(depth_dir, do_flip)

        inputs[("color", 0, 0)] = inputs[("color", 0, 0)].crop(self.box)
        inputs["depth_gt"] = inputs["depth_gt"][180:900,200:1150]
            
        inputs[("color", 0, 0)] = self.resize[0](inputs[("color", 0, 0)])
        inputs[("color", 0, 0)] = self.to_tensor(inputs[("color", 0, 0)])
        
        return inputs
if __name__ == "__main__":
    ds = C3VDDataset(data_path='/mnt/data-hdd2/Beilei/Dataset/C3VD', height=256, width=320, frame_idxs=[0, -1, 1], num_scales=4, is_train=True)
    
    test = ds[0]