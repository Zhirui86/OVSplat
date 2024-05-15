import os
from pathlib import Path
import warnings

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    # from src.model.lseg import get_semantic_generator
    from src.model.model import ModelWrapper
    from src.model.autoencoder.autoencoder import Autoencoder


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,
                                       'resume': "must"})
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",  # save the lastest k ckpt, can do offline test later
        )
    )
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)
    
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    semantic_generator = get_encoder(cfg.model.semantic)
    autoencoder = Autoencoder(input_channels=512)
    decoder = get_decoder(cfg.model.decoder, cfg.dataset)
    
    # for param in semantic_generator.parameters():
    #     param.requires_grad = False
    # for param in encoder.parameters():
    #     param.requires_grad = False
    # for param in autoencoder.parameters():
    #     param.requires_grad = False
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        semantic_generator,
        autoencoder,
        decoder,
        get_losses(cfg.loss),
        step_tracker
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    # 检查load的ckpt是否保存优化器状态
    # ckpt = torch.load(checkpoint_path)
    # if not 'optimizer_states' in ckpt :
    #     model = ModelWrapper(
    #         cfg.optimizer,
    #         cfg.test,
    #         cfg.train,
    #         encoder,
    #         encoder_visualizer,
    #         get_decoder(cfg.model.decoder, cfg.dataset),
    #         get_losses(cfg.loss),
    #         step_tracker
    #     )
    #     model2 = ModelWrapper(
    #         cfg.optimizer,
    #         cfg.test,
    #         cfg.train,
    #         encoder,
    #         encoder_visualizer,
    #         get_decoder(cfg.model.decoder, cfg.dataset),
    #         get_losses(cfg.loss),
    #         step_tracker
    #     )

        # ckpt_saved = torch.load("/data/gyy/mvsplat/outputs/2024-04-07/23-01-01/checkpoints/epoch10998-step110000.ckpt")
        # model.load_state_dict(ckpt_saved['state_dict'], strict=False)
        # optimizer_state_dict = ckpt_saved['optimizer_states'][0]  # 选择第一个优化器状态
        # scheduler_state_dict = ckpt_saved['lr_schedulers'][0]  # 选择第一个学习率调度器状态

        # model2.load_state_dict(ckpt['state_dict'], strict=False)

        # # 将第一个ckpt文件中的优化器状态和学习率调度器状态加载到第二个模型中
        # optimizer = torch.optim.Adam(model2.parameters(), lr=0.001)  # 重新创建优化器
        # optimizer.load_state_dict(optimizer_state_dict)
        # model2.optimizer = optimizer

        # model2_state_dict = model2.state_dict()
        # checkpoint = {
        #     'state_dict': model2_state_dict,
        #     'optimizer_states': [model2.optimizer.state_dict()],
        #     'lr_schedulers': [scheduler_state_dict],
        #     'pytorch-lightning_version': '2.0.0'
        # }

        # torch.save(checkpoint, checkpoint_path)

    # checkpoint_path2 = "checkpoints/demo_e200.ckpt"
    # if checkpoint_path and checkpoint_path2 :
    #     ckpt1 = torch.load(checkpoint_path, map_location='cpu')
    #     ckpt2 = torch.load(checkpoint_path2, map_location='cpu')
    #     model_state_dict1 = ckpt1['state_dict']
    #     model_state_dict2 = ckpt2['state_dict']
    #     model_state_dict2 = {"semantic_generator." + k:v for k,v in model_state_dict2.items()}

    #     current_model_state = model_wrapper.state_dict()
    #     current_model_keys = set(model_wrapper.state_dict().keys())
    #     loaded_keys1 = set(model_state_dict1.keys())
    #     loaded_keys2 = set(model_state_dict2.keys())

    #     # 只保留当前模型存在的键的权重
    #     common_keys1 = current_model_keys & loaded_keys1
    #     filtered_state_dict = {k: v for k, v in model_state_dict1.items() if k in common_keys1}
    #     for k,v in filtered_state_dict.items():
    #         current_model_state[k] = v
    #     # 删去键名称中的semantic_generator
    #     common_keys2 = current_model_keys & loaded_keys2
    #     filtered_state_dict = {k: v for k, v in model_state_dict2.items() if k in common_keys2}
    #     for k,v in filtered_state_dict.items():
    #         current_model_state[k] = v

    #     lr_dict ={
    #         "encoder": 1.5e-5,
    #         "semantic_generator": 1.5e-4,
    #         "autoencoder": 1.5e-3
    #     }
    #     param_groups = {
    #         'encoder': [],
    #         'semantic_generator': [],
    #         'autoencoder': []
    #     }
    #     for name, param in model_wrapper.named_parameters():
    #         if name.startswith('encoder'):
    #             param_groups['encoder'].append(param)
    #         elif name.startswith('semantic_generator'):
    #             param_groups['semantic_generator'].append(param)    
    #         else:
    #             param_groups['autoencoder'].append(param)
    #     params = [
    #         {'params': param_groups['encoder'], 'lr': lr_dict['encoder']},
    #         {'params': param_groups['semantic_generator'], 'lr': lr_dict['semantic_generator']},
    #         {'params': param_groups['autoencoder'], 'lr': lr_dict['autoencoder']}
    #     ]

    #     optimizer = torch.optim.Adam(params, lr=1.5e-5, betas=(0.9, 0.999), weight_decay=0.0001)

    #     warm_up_steps = 2000
    #     warm_up = torch.optim.lr_scheduler.LinearLR(
    #             optimizer,
    #             1 / warm_up_steps,
    #             1,
    #             total_iters=warm_up_steps,
    #     )

    #     checkpoint = {
    #         'state_dict': current_model_state,
    #         'optimizer_states': [optimizer.state_dict()],
    #         'lr_schedulers': [warm_up.state_dict()],
    #         'pytorch-lightning_version': '2.0.0'
    #     }
        
    #     torch.save(checkpoint, "checkpoints/state_saver.ckpt")

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # torch.set_float32_matmul_precision('high')
    train()
