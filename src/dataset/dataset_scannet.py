import random
import time
import os
import numpy as np
import imageio
import cv2
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
import pandas as pd
import sys
from pathlib import Path
from dataclasses import dataclass


sys.path.append("../")
from .dataset import DatasetCfgCommon
from torch.utils.data import IterableDataset
from typing import Literal
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids
from .base_utils import downsample_gaussian_blur
from .shims.crop_shim import apply_crop_shim
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids, loader_resize
from .asset import *
from .semantic_utils import PointSegClassMapping

def set_seed(index,mode):
    if mode == "train":
        np.random.seed((index+int(time.time()))%(2**16))
        random.seed((index+int(time.time()))%(2**16)+1)
        torch.random.manual_seed((index+int(time.time()))%(2**16)+1)
    else:
        np.random.seed(index % (2 ** 16))
        random.seed(index % (2 ** 16) + 1)
        torch.random.manual_seed(index % (2 ** 16) + 1)

@dataclass
class DatasetscannetCfg(DatasetCfgCommon):
    name: Literal["scannet"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True

# only for training
class ScannetTrainDataset(IterableDataset):
    def __init__(self, args, mode, view_sampler, **kwargs):
        if mode == "train":
            self.scene_path_list = np.loadtxt('configs/scannetv2_train_split.txt',dtype=str).tolist()
        elif mode == "val":
            self.scene_path_list = np.loadtxt('configs/scannetv2_val_split.txt',dtype=str).tolist()
        else:
            self.scene_path_list = np.loadtxt('configs/scannetv2_test_split.txt',dtype=str).tolist()

        self.mode = mode
        self.num_source_views = 2
        self.rectify_inplane_rotation = False

        # image_size = 320
        # self.ratio = image_size / 1296
        # self.h, self.w = int(self.ratio*972), int(image_size)
        self.image_size = (192, 256)
        self.ratio = self.image_size[1] / 1296

        all_rgb_files, all_depth_files, all_pose_files, all_label_files, all_intrinsics_files = [],[],[],[],[]
        for i, scene in enumerate(self.scene_path_list):
            scene_path = os.path.join('datasets/' + scene[:-10])
            pose_files = []
            for f in sorted(os.listdir(os.path.join(scene_path, "pose"))):
                path = os.path.join(scene_path, "pose", f)
                pose = np.loadtxt(path)
                if np.isinf(pose).any() or np.isnan(pose).any():
                    continue
                else:
                    pose_files.append(path)
                    
            rgb_files = [f.replace("pose", "color").replace("txt", "jpg") for f in pose_files]
            depth_files = [f.replace("pose", "depth").replace("txt", "png") for f in pose_files]
            intrinsics_files = [
                os.path.join(scene_path, 'intrinsic/intrinsic_color.txt') for f in rgb_files
            ]
            label_files = [f.replace("pose", "label-filt").replace("txt", "png") for f in pose_files]

            all_rgb_files.append(rgb_files)
            all_depth_files.append(depth_files)
            all_label_files.append(label_files)
            all_pose_files.append(pose_files)
            all_intrinsics_files.append(intrinsics_files)

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files, dtype=object)[index]
        self.all_depth_files = np.array(all_depth_files, dtype=object)[index]
        self.all_label_files = np.array(all_label_files, dtype=object)[index]
        self.all_pose_files = np.array(all_pose_files, dtype=object)[index]
        self.all_intrinsics_files = np.array(all_intrinsics_files, dtype=object)[index]

        mapping_file = 'datasets/scannet/scannetv2-labels.combined.tsv'
        mapping_file = pd.read_csv(mapping_file, sep='\t', header=0)
        scan_ids = mapping_file['id'].values
        nyu40_ids = mapping_file['nyu40id'].values
        scan2nyu = np.zeros(max(scan_ids) + 1, dtype=np.int32)
        for i in range(len(scan_ids)):
            scan2nyu[scan_ids[i]] = nyu40_ids[i]
        self.scan2nyu = scan2nyu
        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 14, 16, 24, 28, 33, 34, 36, 39],
            max_cat_id=40
        )


    def pose_inverse(self, pose):
        R = pose[:, :3].T
        t = - R @ pose[:, 3:]
        inversed_pose = np.concatenate([R, t], -1)
        return np.concatenate([inversed_pose, [[0, 0, 0, 1]]])
        # return inversed_pose

    def __len__(self):
        return 9999  # 确保不会中断
    
    def get_data_one_batch(self, idx, nearby_view_id=None):
        self.nearby_view_id = nearby_view_id
        return self.__getitem__(idx=idx)
    
    def normalize_intrinsics(self, intrinsics, img_size):
        h, w = img_size
        # 归一化内参矩阵
        intrinsics_normalized = intrinsics.clone()
        intrinsics_normalized[:, 0, 0] /= w
        intrinsics_normalized[:, 1, 1] /= h
        intrinsics_normalized[:, 0, 2] = 0.5
        intrinsics_normalized[:, 1, 2] = 0.5
        return intrinsics_normalized
    
    def __iter__(self): 
            id_scene = random.randint(0,len(self.all_rgb_files))
            # id_scene = 122
            real_idx = id_scene % len(self.all_rgb_files)

            rgb_files = self.all_rgb_files[real_idx]
            scene = rgb_files[0].split('/')[2]
            depth_files = self.all_depth_files[real_idx]
            pose_files = self.all_pose_files[real_idx]
            label_files = self.all_label_files[real_idx]
            intrinsics_files = self.all_intrinsics_files[real_idx]

            # if self.mode == "train":
            # id_render = random.randint(0,len(rgb_files))
            id_render = np.random.choice(np.arange(len(pose_files)))
            # elif self.mode == "test":
                # if len(id_render) > 10:
                #     id_render = id_render[:10]
            img_idx = rgb_files[id_render][-8:-1]
            img_idx = img_idx.replace(".jp","") if ".jp" in img_idx else img_idx
            img_idx = img_idx.replace("r","") if "r" in img_idx else img_idx
            img_idx = img_idx.replace("o","") if "o" in img_idx else img_idx
            img_idx = img_idx.replace("l","") if "l" in img_idx else img_idx
            img_idx = img_idx.replace("/","") if "/" in img_idx else img_idx


            train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in pose_files], axis=0)
            render_pose = train_poses[id_render]

            subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

            id_feat_pool = get_nearest_pose_ids(
                render_pose,
                train_poses,
                self.num_source_views * subsample_factor,
                tar_id=id_render,
                angular_dist_method="vector",
            )
            id_feat = np.random.choice(id_feat_pool, self.num_source_views, replace=False)

            if id_render in id_feat:
                assert id_render not in id_feat
            # occasionally include input image
            if np.random.choice([0, 1], p=[0.995, 0.005]):
                id_feat[np.random.choice(len(id_feat))] = id_render

            rgb = imageio.imread(rgb_files[id_render]).astype(np.float32) / 255.0
            if self.image_size[1] != 1296:
                rgb = cv2.resize(downsample_gaussian_blur(
                    rgb, self.ratio), (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
                
            img = Image.open(depth_files[id_render])
            depth = np.asarray(img, dtype=np.float32) / 1000.0  # mm -> m
            depth = np.ascontiguousarray(depth, dtype=np.float32)
            depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)

            intrinsics = np.loadtxt(intrinsics_files[id_render]).reshape([4, 4])
            intrinsics[:2, :] *= self.ratio

            img_size = rgb.shape[:2]
            camera = np.concatenate((list(img_size), intrinsics.flatten(), render_pose.flatten())).astype(
                np.float32
            )

            img = Image.open(label_files[id_render])
            label = np.asarray(img, dtype=np.int32)
            label = np.ascontiguousarray(label)
            label = cv2.resize(label, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
            label = label.astype(np.int32)
            label = self.scan2nyu[label]
            label = self.label_mapping(label)

            all_poses = [render_pose]
            # get depth range
            # poses = render_pose[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
            # bds = render_pose[:, -2:].transpose([1, 0])
            # bds = np.moveaxis(bds, -1, 0).astype(np.float32)
            # far_depth = origin_depth + max_radius
            # depth_range = torch.tensor([near_depth, far_depth])
            depth_range = torch.tensor([0.1, 10.0])

            src_rgbs = []
            src_cameras = []
            src_intrinsics = []
            src_extrinsics = []
            src_labels = []
            for id in id_feat:
                src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.0
                
                src_label = Image.open(label_files[id])
                src_label = np.asarray(src_label, dtype=np.int32)
                src_label = np.ascontiguousarray(src_label)
                src_label = cv2.resize(src_label, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
                src_label = src_label.astype(np.int32)
                src_label = self.scan2nyu[src_label]
                src_label = self.label_mapping(src_label)

                if self.image_size[1] != 1296:
                    src_rgb = cv2.resize(downsample_gaussian_blur(
                        src_rgb, self.ratio), (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
                pose = np.loadtxt(pose_files[id]).reshape(4, 4)

                if self.rectify_inplane_rotation:
                    pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

                src_rgbs.append(src_rgb)
                intrinsics = np.loadtxt(intrinsics_files[id]).reshape([4, 4])
                intrinsics[:2, :] *= self.ratio
                img_size = src_rgb.shape[:2]
                src_camera = np.concatenate((list(img_size), intrinsics.flatten(), pose.flatten())).astype(
                    np.float32
                )
                
                src_cameras.append(src_camera)
                src_extrinsics.append(pose)
                src_labels.append(src_label)

            src_rgbs = np.stack(src_rgbs)
            src_cameras = np.stack(src_cameras)
            src_extrinsics = np.stack(src_extrinsics)


            #分界线
            rgb, camera, src_rgbs, src_cameras, intrinsics, src_intrinsics = loader_resize(rgb,camera,src_rgbs,src_cameras, size=self.image_size)
            src_extrinsics = torch.from_numpy(src_extrinsics).float()
            extrinsics = torch.from_numpy(render_pose).unsqueeze(0).float()
            
            src_intrinsics = self.normalize_intrinsics(torch.from_numpy(src_intrinsics[:,:3,:3]).float(), self.image_size)
            intrinsics = self.normalize_intrinsics(torch.from_numpy(intrinsics[:3,:3]).unsqueeze(0).float(), self.image_size)

            depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

            # Resize the world to make the baseline 1.
            if src_extrinsics.shape[0] == 2:
                a, b = src_extrinsics[:, :3, 3]
                scale = (a - b).norm()
                if scale < 0.001:
                    print(
                        f"Skipped {scene} because of insufficient baseline "
                        f"{scale:.6f}"
                    )
                src_extrinsics[:, :3, 3] /= scale
                extrinsics[:, :3, 3] /= scale
            else:
                scale = 1

            example = {
                        "context": {
                                "extrinsics": src_extrinsics,
                                "intrinsics": src_intrinsics,
                                "image": torch.from_numpy(src_rgbs[..., :3]).permute(0, 3, 1, 2),
                                "near":  (depth_range[0].repeat(self.num_source_views) / scale).float(),
                                "far": (depth_range[1].repeat(self.num_source_views) / scale).float(),
                                "index": torch.from_numpy(id_feat),
                                "labels": torch.tensor(src_labels)
                        },                                                                                        
                        "target": {
                                "extrinsics": extrinsics,
                                "intrinsics": intrinsics,
                                "image": torch.from_numpy(rgb[..., :3]).unsqueeze(0).permute(0, 3, 1, 2),
                                "near": (depth_range[0].unsqueeze(0) / scale).float(),
                                "far": (depth_range[1].unsqueeze(0) / scale).float(),
                                "index": torch.tensor([int(img_idx)]),
                                "labels": torch.tensor(label)
                        
                        },"scene":scene}
            yield apply_crop_shim(example, tuple(self.image_size))



# only for validation
# class ScannetValDataset(Dataset):
#     def __init__(self, args, is_train, scenes=None, **kwargs):
#         self.is_train = is_train
#         self.num_source_views = args.num_source_views
#         self.rectify_inplane_rotation = args.rectify_inplane_rotation

#         image_size = 320
#         self.ratio = image_size / 1296
#         self.h, self.w = int(self.ratio*972), int(image_size)

#         scene_path = os.path.join(args.rootdir + 'data', scenes[:-10])
#         pose_files = []
#         for f in sorted(os.listdir(os.path.join(scene_path, "pose"))):
#             path = os.path.join(scene_path, "pose", f)
#             pose = np.loadtxt(path)
#             if np.isinf(pose).any() or np.isnan(pose).any():
#                 continue
#             else:
#                 pose_files.append(path)
                
#         rgb_files = [f.replace("pose", "color").replace("txt", "jpg") for f in pose_files]
#         depth_files = [f.replace("pose", "depth").replace("txt", "png") for f in pose_files]
#         intrinsics_files = [
#             os.path.join(scene_path, 'intrinsic/intrinsic_color.txt') for f in rgb_files
#         ]
#         label_files = [f.replace("pose", "label-filt").replace("txt", "png") for f in pose_files]

#         index = np.arange(len(rgb_files))
#         self.rgb_files = np.array(rgb_files, dtype=object)[index]
#         self.depth_files = np.array(depth_files, dtype=object)[index]
#         self.label_files = np.array(label_files, dtype=object)[index]
#         self.pose_files = np.array(pose_files, dtype=object)[index]
#         self.intrinsics_files = np.array(intrinsics_files, dtype=object)[index]

#         mapping_file = 'data/scannet/scannetv2-labels.combined.tsv'
#         mapping_file = pd.read_csv(mapping_file, sep='\t', header=0)
#         scan_ids = mapping_file['id'].values
#         nyu40_ids = mapping_file['nyu40id'].values
#         scan2nyu = np.zeros(max(scan_ids) + 1, dtype=np.int32)
#         for i in range(len(scan_ids)):
#             scan2nyu[scan_ids[i]] = nyu40_ids[i]
#         self.scan2nyu = scan2nyu
#         self.label_mapping = PointSegClassMapping(
#             valid_cat_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#                            11, 12, 14, 16, 24, 28, 33, 34, 36, 39],
#             max_cat_id=40
#         )

#         que_idxs = np.arange(len(self.rgb_files))
#         self.train_que_idxs = que_idxs[:700:3]
#         self.val_que_idxs = que_idxs[2:700:20]
#         if len(self.val_que_idxs) > 10:
#             self.val_que_idxs = self.val_que_idxs[:10]

#     def __len__(self):
#         if self.is_train is True:
#             return len(self.train_que_idxs)
#         else:  
#             return len(self.val_que_idxs)  
    
#     def __getitem__(self, idx):
#         set_seed(idx, is_train=self.is_train)
#         if self.is_train is True:
#             que_idx = self.train_que_idxs[idx]
#         else:
#             que_idx = self.val_que_idxs[idx]

#         rgb_files = self.rgb_files
#         depth_files = self.depth_files
#         pose_files = self.pose_files
#         label_files = self.label_files
#         intrinsics_files = self.intrinsics_files

#         train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in pose_files], axis=0)
#         render_pose = train_poses[que_idx]

#         subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

#         id_feat_pool = get_nearest_pose_ids(
#             render_pose,
#             train_poses,
#             self.num_source_views * subsample_factor,
#             tar_id=que_idx,
#             angular_dist_method="vector",
#         )
#         id_feat = np.random.choice(id_feat_pool, self.num_source_views, replace=False)

#         if que_idx in id_feat:
#             assert que_idx not in id_feat
#         # occasionally include input image
#         if np.random.choice([0, 1], p=[0.995, 0.005]):
#             id_feat[np.random.choice(len(id_feat))] = que_idx

#         img = Image.open(depth_files[que_idx])
#         depth = np.asarray(img, dtype=np.float32) / 1000.0  # mm -> m
#         depth = np.ascontiguousarray(depth, dtype=np.float32)
#         depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

#         rgb = imageio.imread(rgb_files[que_idx]).astype(np.float32) / 255.0

#         if self.w != 1296:
#             rgb = cv2.resize(downsample_gaussian_blur(
#                 rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            
#         intrinsics = np.loadtxt(intrinsics_files[que_idx]).reshape([4, 4])
#         intrinsics[:2, :] *= self.ratio

#         img_size = rgb.shape[:2]
#         camera = np.concatenate((list(img_size), intrinsics.flatten(), render_pose.flatten())).astype(
#             np.float32
#         )

#         img = Image.open(label_files[que_idx])
#         label = np.asarray(img, dtype=np.int32)
#         label = np.ascontiguousarray(label)
#         label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
#         label = label.astype(np.int32)
#         label = self.scan2nyu[label]
#         label = self.label_mapping(label)

#         all_poses = [render_pose]
#         depth_range = np.array([0.1, 6.0])

#         depth_mask = np.ones_like(depth)
#         depth_mask[depth == 0] = 0

#         src_rgbs = []
#         src_cameras = []
#         for id in id_feat:
#             src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.0
#             if self.w != 1296:
#                 src_rgb = cv2.resize(downsample_gaussian_blur(
#                     src_rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
#             pose = np.loadtxt(pose_files[id]).reshape(4, 4)

#             if self.rectify_inplane_rotation:
#                 pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

#             src_rgbs.append(src_rgb)
#             intrinsics = np.loadtxt(intrinsics_files[id]).reshape([4, 4])
#             intrinsics[:2, :] *= self.ratio
#             img_size = src_rgb.shape[:2]
#             src_camera = np.concatenate((list(img_size), intrinsics.flatten(), pose.flatten())).astype(
#                 np.float32
#             )
#             src_cameras.append(src_camera)
#             all_poses.append(pose)

#         src_rgbs = np.stack(src_rgbs)
#         src_cameras = np.stack(src_cameras)

#         return {
#             "rgb": torch.from_numpy(rgb),
#             "true_depth": torch.from_numpy(depth),
#             "depth_mask": torch.from_numpy(depth_mask),
#             "labels": torch.from_numpy(label),
#             "camera": torch.from_numpy(camera),
#             "rgb_path": rgb_files[que_idx],
#             "src_rgbs": torch.from_numpy(src_rgbs),
#             "src_cameras": torch.from_numpy(src_cameras),
#             "depth_range": torch.from_numpy(depth_range),
#         }


class ScannetTestDataset(IterableDataset):
    def __init__(self, args, mode, view_sampler, **kwargs):
        if mode == "train":
            self.scene_path_list = np.loadtxt('configs/scannetv2_train_split.txt',dtype=str).tolist()
        elif mode == "val":
            self.scene_path_list = np.loadtxt('configs/scannetv2_val_split.txt',dtype=str).tolist()
        else:
            self.scene_path_list = np.loadtxt('configs/scannetv2_test_split.txt',dtype=str).tolist()

        self.mode = mode
        self.num_source_views = 5
        self.rectify_inplane_rotation = False

        # image_size = 320
        # self.ratio = image_size / 1296
        # self.h, self.w = int(self.ratio*972), int(image_size)
        self.image_size = (240, 320)
        self.ratio = self.image_size[1] / 1296

        all_rgb_files, all_depth_files, all_pose_files, all_label_files, all_intrinsics_files = [],[],[],[],[]
        for i, scene in enumerate(self.scene_path_list):
            scene_path = os.path.join('datasets/' + scene[:-10])
            pose_files = []
            for f in sorted(os.listdir(os.path.join(scene_path, "pose"))):
                path = os.path.join(scene_path, "pose", f)
                pose = np.loadtxt(path)
                if np.isinf(pose).any() or np.isnan(pose).any():
                    continue
                else:
                    pose_files.append(path)
                    
            rgb_files = [f.replace("pose", "color").replace("txt", "jpg") for f in pose_files]
            depth_files = [f.replace("pose", "depth").replace("txt", "png") for f in pose_files]
            intrinsics_files = [
                os.path.join(scene_path, 'intrinsic/intrinsic_color.txt') for f in rgb_files
            ]
            label_files = [f.replace("pose", "label-filt").replace("txt", "png") for f in pose_files]

            all_rgb_files.append(rgb_files)
            all_depth_files.append(depth_files)
            all_label_files.append(label_files)
            all_pose_files.append(pose_files)
            all_intrinsics_files.append(intrinsics_files)

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files, dtype=object)[index]
        self.all_depth_files = np.array(all_depth_files, dtype=object)[index]
        self.all_label_files = np.array(all_label_files, dtype=object)[index]
        self.all_pose_files = np.array(all_pose_files, dtype=object)[index]
        self.all_intrinsics_files = np.array(all_intrinsics_files, dtype=object)[index]

        mapping_file = 'datasets/scannet/scannetv2-labels.combined.tsv'
        mapping_file = pd.read_csv(mapping_file, sep='\t', header=0)
        scan_ids = mapping_file['id'].values
        nyu40_ids = mapping_file['nyu40id'].values
        scan2nyu = np.zeros(max(scan_ids) + 1, dtype=np.int32)
        for i in range(len(scan_ids)):
            scan2nyu[scan_ids[i]] = nyu40_ids[i]
        self.scan2nyu = scan2nyu
        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 14, 16, 24, 28, 33, 34, 36, 39],
            max_cat_id=40
        )


    def pose_inverse(self, pose):
        R = pose[:, :3].T
        t = - R @ pose[:, 3:]
        inversed_pose = np.concatenate([R, t], -1)
        return np.concatenate([inversed_pose, [[0, 0, 0, 1]]])
        # return inversed_pose

    def __len__(self):
        return 9999  # 确保不会中断
    
    def get_data_one_batch(self, idx, nearby_view_id=None):
        self.nearby_view_id = nearby_view_id
        return self.__getitem__(idx=idx)
    
    def normalize_intrinsics(self, intrinsics, img_size):
        h, w = img_size
        # 归一化内参矩阵
        intrinsics_normalized = intrinsics.clone()
        intrinsics_normalized[:, 0, 0] /= w
        intrinsics_normalized[:, 1, 1] /= h
        intrinsics_normalized[:, 0, 2] = 0.5
        intrinsics_normalized[:, 1, 2] = 0.5
        return intrinsics_normalized
    
    def __iter__(self): 
        for id_scene in range(0,len(self.all_rgb_files)):
            set_seed(id_scene,self.mode)  
            real_idx = id_scene % len(self.all_rgb_files)
            self.render_list = np.arange(len(self.all_rgb_files[real_idx]))
            self.render_list = self.render_list[2:700:20]
            if len(self.render_list) > 10:
                self.render_list = self.render_list[:10]
            rgb_files = self.all_rgb_files[real_idx]
            scene = rgb_files[0].split('/')[2]
            depth_files = self.all_depth_files[real_idx]
            pose_files = self.all_pose_files[real_idx]
            label_files = self.all_label_files[real_idx]
            intrinsics_files = self.all_intrinsics_files[real_idx]

            # if self.mode == "train":
            for i in range(0,len(self.render_list)):
                id_render = self.render_list[i]
            # elif self.mode == "test":
                # if len(id_render) > 10:
                #     id_render = id_render[:10]
                img_idx = rgb_files[id_render][-8:-1]
                img_idx = img_idx.replace(".jp","") if ".jp" in img_idx else img_idx
                img_idx = img_idx.replace("r","") if "r" in img_idx else img_idx
                img_idx = img_idx.replace("o","") if "o" in img_idx else img_idx
                img_idx = img_idx.replace("l","") if "l" in img_idx else img_idx
                img_idx = img_idx.replace("/","") if "/" in img_idx else img_idx


                train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in pose_files], axis=0)
                render_pose = train_poses[id_render]

                subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

                id_feat_pool = get_nearest_pose_ids(
                    render_pose,
                    train_poses,
                    self.num_source_views * subsample_factor,
                    tar_id=id_render,
                    angular_dist_method="vector",
                )
                id_feat = np.random.choice(id_feat_pool, self.num_source_views, replace=False)

                if id_render in id_feat:
                    assert id_render not in id_feat
                # occasionally include input image
                if np.random.choice([0, 1], p=[0.995, 0.005]):
                    id_feat[np.random.choice(len(id_feat))] = id_render

                rgb = imageio.imread(rgb_files[id_render]).astype(np.float32) / 255.0
                if self.image_size[1] != 1296:
                    rgb = cv2.resize(downsample_gaussian_blur(
                        rgb, self.ratio), (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
                    
                img = Image.open(depth_files[id_render])
                depth = np.asarray(img, dtype=np.float32) / 1000.0  # mm -> m
                depth = np.ascontiguousarray(depth, dtype=np.float32)
                depth = cv2.resize(depth, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)

                intrinsics = np.loadtxt(intrinsics_files[id_render]).reshape([4, 4])
                intrinsics[:2, :] *= self.ratio

                img_size = rgb.shape[:2]
                camera = np.concatenate((list(img_size), intrinsics.flatten(), render_pose.flatten())).astype(
                    np.float32
                )

                img = Image.open(label_files[id_render])
                label = np.asarray(img, dtype=np.int32)
                label = np.ascontiguousarray(label)
                label = cv2.resize(label, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_NEAREST)
                label = label.astype(np.int32)
                label = self.scan2nyu[label]
                label = self.label_mapping(label)

                all_poses = [render_pose]
                # get depth range
                # poses = render_pose[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
                # bds = render_pose[:, -2:].transpose([1, 0])
                # bds = np.moveaxis(bds, -1, 0).astype(np.float32)
                # far_depth = origin_depth + max_radius
                # depth_range = torch.tensor([near_depth, far_depth])
                depth_range = torch.tensor([0.1, 10.0])

                src_rgbs = []
                src_cameras = []
                src_intrinsics = []
                src_extrinsics = []
                for id in id_feat:
                    src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.0
                    if self.image_size[1] != 1296:
                        src_rgb = cv2.resize(downsample_gaussian_blur(
                            src_rgb, self.ratio), (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
                    pose = np.loadtxt(pose_files[id]).reshape(4, 4)

                    if self.rectify_inplane_rotation:
                        pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

                    src_rgbs.append(src_rgb)
                    intrinsics = np.loadtxt(intrinsics_files[id]).reshape([4, 4])
                    intrinsics[:2, :] *= self.ratio
                    img_size = src_rgb.shape[:2]
                    src_camera = np.concatenate((list(img_size), intrinsics.flatten(), pose.flatten())).astype(
                        np.float32
                    )
                    
                    src_cameras.append(src_camera)
                    src_extrinsics.append(pose)

                src_rgbs = np.stack(src_rgbs)
                src_cameras = np.stack(src_cameras)
                src_extrinsics = np.stack(src_extrinsics)


                #分界线
                rgb, camera, src_rgbs, src_cameras, intrinsics, src_intrinsics = loader_resize(rgb,camera,src_rgbs,src_cameras, size=self.image_size)
                src_extrinsics = torch.from_numpy(src_extrinsics).float()
                extrinsics = torch.from_numpy(render_pose).unsqueeze(0).float()
                
                src_intrinsics = self.normalize_intrinsics(torch.from_numpy(src_intrinsics[:,:3,:3]).float(), self.image_size)
                intrinsics = self.normalize_intrinsics(torch.from_numpy(intrinsics[:3,:3]).unsqueeze(0).float(), self.image_size)

                depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

                # Resize the world to make the baseline 1.
                if src_extrinsics.shape[0] == 2:
                    a, b = src_extrinsics[:, :3, 3]
                    scale = (a - b).norm()
                    if scale < 0.001:
                        print(
                            f"Skipped {scene} because of insufficient baseline "
                            f"{scale:.6f}"
                        )
                    src_extrinsics[:, :3, 3] /= scale
                    extrinsics[:, :3, 3] /= scale
                else:
                    scale = 1

                example = {
                            "context": {
                                    "extrinsics": src_extrinsics,
                                    "intrinsics": src_intrinsics,
                                    "image": torch.from_numpy(src_rgbs[..., :3]).permute(0, 3, 1, 2),
                                    "near":  (depth_range[0].repeat(self.num_source_views) / scale).float(),
                                    "far": (depth_range[1].repeat(self.num_source_views) / scale).float(),
                                    "index": torch.from_numpy(id_feat),
                            },                                                                                        
                            "target": {
                                    "extrinsics": extrinsics,
                                    "intrinsics": intrinsics,
                                    "image": torch.from_numpy(rgb[..., :3]).unsqueeze(0).permute(0, 3, 1, 2),
                                    "near": (depth_range[0].unsqueeze(0) / scale).float(),
                                    "far": (depth_range[1].unsqueeze(0) / scale).float(),
                                    "index": torch.tensor([int(img_idx)]),
                            
                            },"scene":scene}
                yield apply_crop_shim(example, tuple(self.image_size))



# only for validation
# class ScannetValDataset(Dataset):
#     def __init__(self, args, is_train, scenes=None, **kwargs):
#         self.is_train = is_train
#         self.num_source_views = args.num_source_views
#         self.rectify_inplane_rotation = args.rectify_inplane_rotation

#         image_size = 320
#         self.ratio = image_size / 1296
#         self.h, self.w = int(self.ratio*972), int(image_size)

#         scene_path = os.path.join(args.rootdir + 'data', scenes[:-10])
#         pose_files = []
#         for f in sorted(os.listdir(os.path.join(scene_path, "pose"))):
#             path = os.path.join(scene_path, "pose", f)
#             pose = np.loadtxt(path)
#             if np.isinf(pose).any() or np.isnan(pose).any():
#                 continue
#             else:
#                 pose_files.append(path)
                
#         rgb_files = [f.replace("pose", "color").replace("txt", "jpg") for f in pose_files]
#         depth_files = [f.replace("pose", "depth").replace("txt", "png") for f in pose_files]
#         intrinsics_files = [
#             os.path.join(scene_path, 'intrinsic/intrinsic_color.txt') for f in rgb_files
#         ]
#         label_files = [f.replace("pose", "label-filt").replace("txt", "png") for f in pose_files]

#         index = np.arange(len(rgb_files))
#         self.rgb_files = np.array(rgb_files, dtype=object)[index]
#         self.depth_files = np.array(depth_files, dtype=object)[index]
#         self.label_files = np.array(label_files, dtype=object)[index]
#         self.pose_files = np.array(pose_files, dtype=object)[index]
#         self.intrinsics_files = np.array(intrinsics_files, dtype=object)[index]

#         mapping_file = 'data/scannet/scannetv2-labels.combined.tsv'
#         mapping_file = pd.read_csv(mapping_file, sep='\t', header=0)
#         scan_ids = mapping_file['id'].values
#         nyu40_ids = mapping_file['nyu40id'].values
#         scan2nyu = np.zeros(max(scan_ids) + 1, dtype=np.int32)
#         for i in range(len(scan_ids)):
#             scan2nyu[scan_ids[i]] = nyu40_ids[i]
#         self.scan2nyu = scan2nyu
#         self.label_mapping = PointSegClassMapping(
#             valid_cat_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#                            11, 12, 14, 16, 24, 28, 33, 34, 36, 39],
#             max_cat_id=40
#         )

#         que_idxs = np.arange(len(self.rgb_files))
#         self.train_que_idxs = que_idxs[:700:3]
#         self.val_que_idxs = que_idxs[2:700:20]
#         if len(self.val_que_idxs) > 10:
#             self.val_que_idxs = self.val_que_idxs[:10]

#     def __len__(self):
#         if self.is_train is True:
#             return len(self.train_que_idxs)
#         else:  
#             return len(self.val_que_idxs)  
    
#     def __getitem__(self, idx):
#         set_seed(idx, is_train=self.is_train)
#         if self.is_train is True:
#             que_idx = self.train_que_idxs[idx]
#         else:
#             que_idx = self.val_que_idxs[idx]

#         rgb_files = self.rgb_files
#         depth_files = self.depth_files
#         pose_files = self.pose_files
#         label_files = self.label_files
#         intrinsics_files = self.intrinsics_files

#         train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in pose_files], axis=0)
#         render_pose = train_poses[que_idx]

#         subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

#         id_feat_pool = get_nearest_pose_ids(
#             render_pose,
#             train_poses,
#             self.num_source_views * subsample_factor,
#             tar_id=que_idx,
#             angular_dist_method="vector",
#         )
#         id_feat = np.random.choice(id_feat_pool, self.num_source_views, replace=False)

#         if que_idx in id_feat:
#             assert que_idx not in id_feat
#         # occasionally include input image
#         if np.random.choice([0, 1], p=[0.995, 0.005]):
#             id_feat[np.random.choice(len(id_feat))] = que_idx

#         img = Image.open(depth_files[que_idx])
#         depth = np.asarray(img, dtype=np.float32) / 1000.0  # mm -> m
#         depth = np.ascontiguousarray(depth, dtype=np.float32)
#         depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

#         rgb = imageio.imread(rgb_files[que_idx]).astype(np.float32) / 255.0

#         if self.w != 1296:
#             rgb = cv2.resize(downsample_gaussian_blur(
#                 rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            
#         intrinsics = np.loadtxt(intrinsics_files[que_idx]).reshape([4, 4])
#         intrinsics[:2, :] *= self.ratio

#         img_size = rgb.shape[:2]
#         camera = np.concatenate((list(img_size), intrinsics.flatten(), render_pose.flatten())).astype(
#             np.float32
#         )

#         img = Image.open(label_files[que_idx])
#         label = np.asarray(img, dtype=np.int32)
#         label = np.ascontiguousarray(label)
#         label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
#         label = label.astype(np.int32)
#         label = self.scan2nyu[label]
#         label = self.label_mapping(label)

#         all_poses = [render_pose]
#         depth_range = np.array([0.1, 6.0])

#         depth_mask = np.ones_like(depth)
#         depth_mask[depth == 0] = 0

#         src_rgbs = []
#         src_cameras = []
#         for id in id_feat:
#             src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.0
#             if self.w != 1296:
#                 src_rgb = cv2.resize(downsample_gaussian_blur(
#                     src_rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
#             pose = np.loadtxt(pose_files[id]).reshape(4, 4)

#             if self.rectify_inplane_rotation:
#                 pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

#             src_rgbs.append(src_rgb)
#             intrinsics = np.loadtxt(intrinsics_files[id]).reshape([4, 4])
#             intrinsics[:2, :] *= self.ratio
#             img_size = src_rgb.shape[:2]
#             src_camera = np.concatenate((list(img_size), intrinsics.flatten(), pose.flatten())).astype(
#                 np.float32
#             )
#             src_cameras.append(src_camera)
#             all_poses.append(pose)

#         src_rgbs = np.stack(src_rgbs)
#         src_cameras = np.stack(src_cameras)

#         return {
#             "rgb": torch.from_numpy(rgb),
#             "true_depth": torch.from_numpy(depth),
#             "depth_mask": torch.from_numpy(depth_mask),
#             "labels": torch.from_numpy(label),
#             "camera": torch.from_numpy(camera),
#             "rgb_path": rgb_files[que_idx],
#             "src_rgbs": torch.from_numpy(src_rgbs),
#             "src_cameras": torch.from_numpy(src_cameras),
#             "depth_range": torch.from_numpy(depth_range),
#         }

