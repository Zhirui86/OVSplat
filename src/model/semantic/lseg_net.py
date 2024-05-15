import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lseg_blocks import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit
import clip
import numpy as np
import pandas as pd
import os

class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

def _make_fusion_block(features, use_bn, fusion):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        fusion=fusion
    )

class LSeg(BaseModel):
    def __init__(
        self,
        labels,
        arch_option,
        block_depth,
        activation,
        scale_factor=0.5, 
        crop_size=480,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        channels_last=False,
        use_bn=True,
    ):
        super(LSeg, self).__init__()
        self.arch_option = arch_option
        self.block_depth = block_depth
        self.activation = activation
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = labels
        self.channels_last = channels_last
        self.head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }
            
        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )
        
        for params in self.clip_pretrained.parameters():
            params.requires_grad = False
        # for params in self.pretrained.parameters():
        #     params.requires_grad = False
        # for params in self.scratch.parameters():
        #     params.requires_grad = False
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn, fusion=True)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn, fusion=True)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, fusion=True)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn, fusion=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation=self.activation)
            self.block_depth = self.block_depth
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation=self.activation)
            self.block_depth = self.block_depth

        self.scratch.output_conv = self.head

        self.text = clip.tokenize(self.labels)    
        self.eps = 1e-8
        
    def forward(self, x, image_features=None, labelset=''):
        if labelset == '':
            text = self.text
            text = text.to(x.device)
            self.logit_scale = self.logit_scale.to(x.device)
            text_features = self.clip_pretrained.encode_text(text)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + self.eps)
        else:
            text = clip.tokenize(labelset)    
            text = text.to(x.device)
            self.logit_scale = self.logit_scale.to(x.device)
            text_features = self.clip_pretrained.encode_text(text)
            text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + self.eps)
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)
        if image_features == None:
            layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

            layer_1_rn = self.scratch.layer1_rn(layer_1)
            layer_2_rn = self.scratch.layer2_rn(layer_2)
            layer_3_rn = self.scratch.layer3_rn(layer_3)
            layer_4_rn = self.scratch.layer4_rn(layer_4)

            path_4 = self.scratch.refinenet4(layer_4_rn)
            path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
            path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
            path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

            image_features = self.scratch.head1(path_1)
            imshape = image_features.shape
            image_features = image_features.permute(0,2,3,1).reshape(-1, self.out_c)
            # normalized features
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + self.eps)
        else:
            imshape = image_features.shape
            image_features = image_features.squeeze(0).permute(1,2,0).view(-1,512)
            image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + self.eps)
            
        logits_per_image = self.logit_scale * image_features.half() @ text_features.t()

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

        if self.arch_option in [1, 2]:
            for _ in range(self.block_depth - 1):
                out = self.scratch.head_block(out)
            out = self.scratch.head_block(out, False)

        # out = self.scratch.output_conv(out)           
        return out, image_features


class LSegNet(LSeg):
    """Network for semantic segmentation."""
    def __init__(self, labels, backbone, features, arch_option, block_depth, activation, path=None, scale_factor=0.5, crop_size=480):

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = labels

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, cfg)

        if path is not None:
            self.load(path)


    
        
    