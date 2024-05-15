import re
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from argparse import ArgumentParser
import pytorch_lightning as pl
from .lsegmentation_module import LSegmentationModule
from .lseg_net import LSeg
from encoding.models.sseg.base import up_kwargs
from dataclasses import dataclass
from typing import Literal
from ..encoder import Encoder
from encoding.nn import SegmentationLosses



import os
import clip
import numpy as np

from scipy import signal
import glob

from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

@dataclass
class LSegModuleCfg:
    name: Literal["semantic_generator"]
    backbone: str
    num_features: int
    dropout: float
    no_scaleinv: bool
    no_batchnorm: bool
    widehead: bool
    arch_option: int
    block_depth: int
    activation: str
    aux: bool
    aux_weight: float
    se_loss: bool
    se_weight: float
    ignore_index: int
    use_bn: bool


class LSegModule(Encoder[LSegModuleCfg]):
    def __init__(self, cfg: LSegModuleCfg):
        super().__init__(cfg)

        # if dataset == "citys":
        #     self.base_size = 2048
        #     self.crop_size = 768
        # else:
        self.base_size = 520
        self.crop_size = 480

        use_pretrained = True
        norm_mean= [0.5, 0.5, 0.5]
        norm_std = [0.5, 0.5, 0.5]

        print('** Use norm {}, {} as the mean and std **'.format(norm_mean, norm_std))

        train_transform = [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]

        val_transform = [
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ]

        self.train_transform = transforms.Compose(train_transform)
        self.val_transform = transforms.Compose(val_transform)

        # self.trainset = self.get_trainset(
        #     dataset,
        #     augment=kwargs["augment"],
        #     base_size=self.base_size,
        #     crop_size=self.crop_size,
        # )
        
        # self.valset = self.get_valset(
        #     dataset,
        #     augment=kwargs["augment"],
        #     base_size=self.base_size,
        #     crop_size=self.crop_size,
        # )

        # use_batchnorm = (
        #     (not cfg.no_batchnorm) if "no_batchnorm" in cfg else True
        # )
        # print(kwargs)

        labels = self.get_labels('scannet')
        self.num_classes = len(labels)

        self.net = LSeg(
            labels=labels,
            arch_option=cfg.arch_option,
            block_depth=cfg.block_depth,
            activation=cfg.activation,
            backbone=cfg.backbone,
            features=cfg.num_features,
            crop_size=self.crop_size,
        )

        self.net.pretrained.model.patch_embed.img_size = (
            self.crop_size,
            self.crop_size,
        )

        self._up_kwargs = up_kwargs
        self.mean = norm_mean
        self.std = norm_std

        self.criterion = self.get_criterion(cfg)

    def get_criterion(self, cfg):
        return SegmentationLosses(
            se_loss=cfg.se_loss, 
            aux=cfg.aux, 
            nclass=self.num_classes, 
            se_weight=cfg.se_weight, 
            aux_weight=cfg.aux_weight, 
            ignore_index=cfg.ignore_index, 
        )
                                
    def get_labels(self, dataset):
        labels = ["wall","floor","cabinet","bed","chair","sofa","table","door",
                  "window","bookshelf","picture","counter","desk","curtain",
                  "refrigerator","shower curtain","toilet","sink","bathtub",
                  "otherfurniture"]
        # path = "/data/gyy/mvsplat/datasets/scannetv2-labels.combined.tsv"
        # # assert os.path.exists(path), '*** Error : {} not exist !!!'.format(path)
        # f = open(path, 'r') 
        # lines = f.readlines()      
        # for line in lines: 
        #     label = line.strip().split('\t')[7]
        #     labels.append(label)
        # f.close()
        # if dataset in ['scannet']:
        #     labels = labels[1:]
        return labels
    
    def forward(self, 
                x,image_feature):
        return self.net(x,image_feature)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LSegmentationModule.add_model_specific_args(parent_parser)
        parser = ArgumentParser(parents=[parser])

        parser.add_argument(
            "--backbone",
            type=str,
            default="clip_vitl16_384",
            help="backbone network",
        )

        parser.add_argument(
            "--num_features",
            type=int,
            default=256,
            help="number of featurs that go from encoder to decoder",
        )

        parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")

        parser.add_argument(
            "--finetune_weights", type=str, help="load weights to finetune from"
        )

        parser.add_argument(
            "--no-scaleinv",
            default=True,
            action="store_false",
            help="turn off scaleinv layers",
        )

        parser.add_argument(
            "--no-batchnorm",
            default=False,
            action="store_true",
            help="turn off batchnorm",
        )

        parser.add_argument(
            "--widehead", default=False, action="store_true", help="wider output head"
        )

        parser.add_argument(
            "--widehead_hr",
            default=False,
            action="store_true",
            help="wider output head",
        )

        parser.add_argument(
            "--arch_option",
            type=int,
            default=0,
            help="which kind of architecture to be used",
        )

        parser.add_argument(
            "--block_depth",
            type=int,
            default=0,
            help="how many blocks should be used",
        )

        parser.add_argument(
            "--activation",
            choices=['lrelu', 'tanh'],
            default="lrelu",
            help="use which activation to activate the block",
        )

        return parser
