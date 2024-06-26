#!/usr/bin/env bash

# --- Ablation Models ---

#shell for test
CUDA_VISIBLE_DEVICES=3 ~/miniconda/envs/splat/bin/python -m src.main +experiment=scannet \
checkpointing.load=/data/gyy/mvsplat/outputs/2024-04-09/23-01-16/checkpoints/epoch24997-step250000.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true 

#shell for train
CUDA_VISIBLE_DEVICES=1 ~/miniconda/envs/splat/bin/python -m src.main +experiment=scannet \
checkpointing.load=/data/gyy/lsplat/outputs/2024-05-15/18-22-06/checkpoints/1wstepbyonecycle.ckpt \
mode=train \
data_loader.train.batch_size=1


# Table 3: w/o cost volume
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine_wocv.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wocv \
model.encoder.wo_depth_refine=true \
model.encoder.wo_cost_volume=true

# Table 3: w/o cross-view attention
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine_wobbcrossattn_best.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wo_backbone_cross_attn \
model.encoder.wo_depth_refine=true \
model.encoder.wo_backbone_cross_attn=true

# Table 3: w/o U-Net
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine_wounet.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wo_unet \
model.encoder.wo_depth_refine=true \
model.encoder.wo_cost_volume_refine=true

# Table B: w/ Epipolar Transformer
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_worefine_wepitrans.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_w_epipolar_trans \
model.encoder.wo_depth_refine=true \
model.encoder.use_epipolar_trans=true

# Table C: 3 Gaussians per pixel
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_gpp3.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_gpp3 \
model.encoder.gaussians_per_pixel=3

# Table D: w/ random init (300K)
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_wopretrained.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wo_pretrained 

# Table D: w/ random init (450K)
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/ablations/re10k_wopretrained_450k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true \
wandb.name=abl/re10k_wo_pretrained_450k 

# --- Default Final Models ---

# Table 1: re10k
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
test.compute_scores=true

# Table 1: acid
python -m src.main +experiment=acid \
checkpointing.load=checkpoints/acid.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_acid.json \
test.compute_scores=true

# generate video
python -m src.main +experiment=re10k \
checkpointing.load=checkpoints/re10k.ckpt \
mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_re10k_video.json \
test.save_video=true \
test.save_image=false \
test.compute_scores=false
