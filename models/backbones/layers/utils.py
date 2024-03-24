# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import warnings
from typing import Callable
import fvcore.nn.weight_init as weight_init

import torch
import torch.nn.functional as F
import torch.nn as nn

def resize(input, size=None, scale_factor=None, mode="nearest", align_corners=None, warning=False):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if (
                    (output_h > 1 and output_w > 1 and input_h > 1 and input_w > 1)
                    and (output_h - 1) % (input_h - 1)
                    and (output_w - 1) % (input_w - 1)
                ):
                    warnings.warn(
                        f"When align_corners={align_corners}, "
                        "the output would more aligned if "
                        f"input size {(input_h, input_w)} is `x+1` and "
                        f"out size {(output_h, output_w)} is `nx+1`"
                    )
    return F.interpolate(input, size, scale_factor, mode, align_corners)

class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, use_act=True):
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size, stride, padding, dilation)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self.use_act:
            out = self.nonlin(out)
        return out

class ConvBlock_PreAct(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, 
                 in_channels: any, 
                 out_channels, 
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 use_act=True,
                 ):
        super(ConvBlock_PreAct, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), kernel_size, stride, padding, dilation, groups)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.use_act:
            out = self.nonlin(x)
        out = self.conv(out)

        return out
    
class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out
    
class ResBottleneckBlock(nn.Module):
    """
    The standard bottleneck residual block without the last activation layer.
    It contains 3 conv layers with kernels 1x1, 3x3, 1x1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_channels: int,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        conv_kernels=3,
        conv_paddings=1,
    ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            bottleneck_channels (int): number of output channels for the 3x3
                "bottleneck" conv layers.
            norm (str or callable): normalization for all conv layers.
                See :func:`layers.get_norm` for supported format.
            act_layer (callable): activation for all conv layers.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.norm1 = LayerNorm(bottleneck_channels)
        self.act1 = act_layer()
        
        self.conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            conv_kernels,
            padding=conv_paddings,
            bias=False,
        )
        self.norm2 = LayerNorm(bottleneck_channels)
        self.act2 = act_layer()
        
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.norm3 = LayerNorm(out_channels)
        
        for layer in [self.conv1, self.conv2, self.conv3]:
            weight_init.c2_msra_fill(layer)
        for layer in [self.norm1, self.norm2]:
            layer.weight.data.fill_(1.0)
            layer.bias.data.zero_()
        # zero init last norm layer.
        self.norm3.weight.data.zero_()
        self.norm3.bias.data.zero_()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.norm3(out)
        return out
    

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x