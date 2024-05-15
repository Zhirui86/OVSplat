import numpy as np
import torch

def read_class_mapping(mapping_file):
    class_mapping = {}
    with open(mapping_file, 'r') as f:
        next(f)  
        for line in f:
            parts = line.strip().split('\t')
            nyu40_id = int(parts[4])
            nyu40_class = str(parts[7])  
            class_mapping[nyu40_id] = nyu40_class
    return class_mapping

def map_labels(label_image, class_mapping):
    # class_mapping_dict = {k: torch.tensor(v) for k, v in class_mapping.items()}
    
    # 创建一个新的张量来映射标签
    mapped_labels = torch.zeros_like(label_image)
    for k, v in class_mapping.items():
        mapped_labels[label_image == k] = v

    # 将张量展平并转换为列表
    flat_labels = mapped_labels.flatten()
    labels = list(set(flat_labels.numpy()))  # 转换回 NumPy 数组再转换为列表，以去除重复项
    return labels, mapped_labels

def reverse_mapping(label_image):
    mapping_file = 'datasets/scannet/scannetv2-labels.combined.tsv'
    class_mapping = read_class_mapping(mapping_file)
    labels, mapped_labels = map_labels(label_image, class_mapping)

    return labels, mapped_labels

def generate_mask(mapped_labels ,labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]):
    mask_dict = {}
    for label in labels:
        mask = (mapped_labels == label)
        mask_dict[label] = mask.float().unsqueeze(1)
    mask = torch.cat(list(mask_dict.values()), dim=1)
    return mask
