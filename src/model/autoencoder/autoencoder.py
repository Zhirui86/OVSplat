import torch
import torch.nn as nn
from typing import Literal, List
from dataclasses import dataclass
from ..encoder.costvolume.ldm_unet.unet import UNetModel


@dataclass
class AutoencoderCfg:
    name: Literal["autoencoder"]
    encoder_hidden_dims: List[int]
    decoder_hidden_dims: List[int]

class Autoencoder(nn.Module):
    def __init__(self, input_channels):
        super(Autoencoder, self).__init__()
        self.input_channels = input_channels
        self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 288*2, 3, 1, 1),
                nn.BatchNorm2d(288*2),
                nn.Conv2d(288*2, 288, 3, 1, 1)
        )

             
        # self.decoder = nn.Sequential(
        #         nn.Conv2d(32, 64, 3, 1, 1),
        #         nn.GroupNorm(4, 64),
        #         nn.GELU(),
        #         nn.Conv2d(64, 128, 3, 1, 1),
        #         nn.GroupNorm(4, 128),
        #         nn.GELU(),
        #         nn.Conv2d(128, 256, 3, 1, 1),
        #         nn.GroupNorm(4, 256),
        #         nn.GELU(),
        #         nn.Conv2d(256, 512, 3, 1, 1),
        #     )
        self.decoder = nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 512, 3, 1, 1),
        )


    def forward(self, x):
        x = self.encoder(x)
        # x = x / x.norm(dim=(2, 3), keepdim=True)
        x = self.decoder(x)
        # x = x / x.norm(dim=(2, 3), keepdim=True)
        return x

    def encode(self, x):
        x = self.encoder(x)
        # x = x / x.norm(dim=(2, 3), keepdim=True)
        return x

    def decode(self, x):
        x = self.decoder(x)
        # x = x / x.norm(dim=(2, 3), keepdim=True)
        return x
