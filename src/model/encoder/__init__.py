from typing import Optional

from .encoder import Encoder
from .encoder_costvolume import EncoderCostVolume, EncoderCostVolumeCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_costvolume import EncoderVisualizerCostVolume
from ..semantic.semantic_generator import LSegModule, LSegModuleCfg
from ..autoencoder.autoencoder import AutoencoderCfg


ENCODERS = {
    "costvolume": (EncoderCostVolume, EncoderVisualizerCostVolume),
    "semantic_generator": LSegModule
}

EncoderCfg = EncoderCostVolumeCfg
SemanticCfg = LSegModuleCfg
autoencodercfg = AutoencoderCfg



def get_encoder(cfg) :
    if cfg.name == "costvolume":
        encoder, visualizer = ENCODERS[cfg.name]
        encoder = encoder(cfg)
        if visualizer is not None:
            visualizer = visualizer(cfg.visualizer, encoder)
        return encoder, visualizer
        
    else:
        encoder = ENCODERS[cfg.name]
        encoder = encoder(cfg)
        return encoder
