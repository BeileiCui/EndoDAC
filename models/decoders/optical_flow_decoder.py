from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from torch.distributions.normal import Normal
from collections import OrderedDict
from utils.layers import *


class PositionDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales = range(4) , num_output_channels=2, use_skips=True):
        super(PositionDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.conv = getattr(nn, 'Conv2d')

        # decoder
        self.convs = OrderedDict() # 有序字典
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:

            self.convs[("position_conv", s)] = self.conv (self.num_ch_dec[s], self.num_output_channels, kernel_size = 3, padding = 1)
            # init flow layer with small weights and bias
            self.convs[("position_conv", s)].weight = nn.Parameter(Normal(0, 1e-5).sample(self.convs[("position_conv", s)].weight.shape))
            self.convs[("position_conv", s)].bias = nn.Parameter(torch.zeros(self.convs[("position_conv", s)].bias.shape))

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        self.outputs = {}
        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("position", i)] = self.convs[("position_conv", i)](x)

        return self.outputs
