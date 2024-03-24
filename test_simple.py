from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2
import torch
from torchvision import transforms, datasets

from utils.layers import disp_to_depth
from options import str2bool
import models.encoders as encoders
import models.decoders as decoders
import models.endodac as endodac

file_dir = os.path.dirname(__file__)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_path', type=str,
                        help='path to the test model', required=True)
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="png")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pretrained_path",
                        type=str,
                        help="pretrained weights path",
                        default=os.path.join(file_dir, "pretrained_model"))
    parser.add_argument("--lora_rank",
                        type=int,
                        help="the rank of lora",
                        default=4)
    parser.add_argument("--lora_type",
                        type=str,
                        help="which lora type use for the model",
                        choices=["lora", "dvlora", "none"],
                        default="dvlora")
    parser.add_argument("--residual_block_indexes",
                        nargs="*",
                        type=int,
                        help="indexes for residual blocks in vitendodepth encoder",
                        default=[2, 5, 8, 11])
    parser.add_argument("--include_cls_token",
                        type=str2bool,
                        help="includes the cls token in the transformer blocks",
                        default=True)
    parser.add_argument("--model_type",
                        type=str,
                        help="which training split to use",
                        choices=["endodac", "afsfm"],
                        default="endodac")
    return parser.parse_args()


def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_path = args.model_path

    print("-> Loading model from ", model_path)
    if args.model_type == 'endodac':
        depther_path = os.path.join(model_path, "depth_model.pth")
        depther_dict = torch.load(depther_path)
        feed_height = depther_dict['height']
        feed_width = depther_dict['width']
    elif args.model_type == 'afsfm':
        encoder_path = os.path.join(model_path, "encoder.pth")
        decoder_path = os.path.join(model_path, "depth.pth")
        encoder_dict = torch.load(encoder_path)
        feed_height = encoder_dict['height']
        feed_width = encoder_dict['width']

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained depther")
    if args.model_type == 'endodac':
        depther = endodac.endodac(
            backbone_size = "base", r=args.lora_rank, lora_type=args.lora_type,
            image_shape=(224,280), pretrained_path=args.pretrained_path,
            residual_block_indexes=args.residual_block_indexes,
            include_cls_token=args.include_cls_token)
        model_dict = depther.state_dict()
        depther.load_state_dict({k: v for k, v in depther_dict.items() if k in model_dict})
        depther.cuda()
        depther.eval()
    elif args.model_type == 'afsfm':
        encoder = encoders.ResnetEncoder(18, False)
        depth_decoder = decoders.DepthDecoder(encoder.num_ch_enc, scales=range(4))
        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))
        depther = lambda image: depth_decoder(encoder(image))
        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')

            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            outputs = depther(input_image)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height * 2, original_width * 2), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 150)
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)

            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax) # 归一化到0-1
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma') # colormap
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)

            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}.jpeg".format(output_name))
            im.save(name_dest_im, quality=95)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print('->p Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
