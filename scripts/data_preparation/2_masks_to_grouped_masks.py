# ------------------------------------------------------------------------
# Copyright (c) 2023 Siwei Li.
# ------------------------------------------------------------------------
import os
import lmdb
import torch
import io
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np

lmdb_dir = "../../datasets/GoPro/test/input_mask.lmdb"
output_folder = "../../datasets/GoPro/test/grouped_masks"

def grayscale_to_masks(grayscale_img):
    unique_values = torch.unique(grayscale_img)
    masks = []
    for value in unique_values:
        mask = (grayscale_img == value).bool()
        masks.append(mask)
    return torch.stack(masks)

def read_image_from_lmdb(lmdb_env, key):
    with lmdb_env.begin() as txn:
        buffer = txn.get(key.encode())
        buffer = io.BytesIO(buffer)
        img = Image.open(buffer).convert('L')
        return ToTensor()(img)

def save_masks_to_folder(folder_path, key, masks):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{key}.npz")
    np.savez_compressed(file_path, masks=masks.numpy())

env = lmdb.open(lmdb_dir, readonly=True, max_dbs=0)

with env.begin() as txn:
    cursor = txn.cursor()
    for key, _ in cursor:
        grayscale_img = read_image_from_lmdb(env, key.decode())
        masks = grayscale_to_masks(grayscale_img)
        save_masks_to_folder(output_folder, key.decode(), masks)

env.close()
