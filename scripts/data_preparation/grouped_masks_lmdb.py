# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

from basicsr.utils import scandir
from basicsr.utils.create_lmdb import create_lmdb_for_gopro, make_lmdb_from_imgs
import argparse
from os import path as osp

def prepare_keys(folder_path, suffix='png'):
    """Prepare image path list and keys for DIV2K dataset.

    Args:
        folder_path (str): Folder path.

    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix=suffix, recursive=False)))
    keys = [img_path.split('.{}'.format(suffix))[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

def main():
    opt = {}
    opt['save_folder'] = '/home/henryli/data/RHM250/test/grouped_masks'

    create_lmdb_for_gopro(opt['save_folder'])

def create_lmdb_for_gopro(folder_path):
    lmdb_path = folder_path + '.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'npz')
    make_lmdb_from_npzs(folder_path, lmdb_path, img_path_list, keys)

import lmdb
from tqdm import tqdm

def read_npz_worker(npz_path, key='masks'):
    """Read npz file and return the array."""
    with open(npz_path, 'rb') as f:
        npz_file = np.load(npz_path)
        return f.read(), npz_file[key].shape

def make_lmdb_from_npzs(data_path,
                        lmdb_path,
                        npz_path_list,
                        keys,
                        batch=5000,
                        map_size=None):
    assert len(npz_path_list) == len(keys), (
        'npz_path_list and keys should have the same length, '
        f'but got {len(npz_path_list)} and {len(keys)}')
    print(f'Create lmdb for {data_path}, save to {lmdb_path}...')
    print(f'Total npz files: {len(npz_path_list)}')
    if not lmdb_path.endswith('.lmdb'):
        raise ValueError("lmdb_path must end with '.lmdb'.")
    if osp.exists(lmdb_path):
        print(f'Folder {lmdb_path} already exists. Exit.')
        sys.exit(1)

    # Create lmdb environment
    if map_size is None:
        map_size = 1024 ** 4  # 1 TB

    env = lmdb.open(lmdb_path, map_size=map_size)

    # Write data to lmdb
    pbar = tqdm(total=len(npz_path_list), unit='chunk')
    txn = env.begin(write=True)
    txt_file = open(osp.join(lmdb_path, 'meta_info.txt'), 'w')
    for idx, (path, key) in enumerate(zip(npz_path_list, keys)):
        pbar.update(1)
        pbar.set_description(f'Write {key}')
        key_byte = key.encode('ascii')
        array_byte, array_shape = read_npz_worker(osp.join(data_path, path))
        txn.put(key_byte, array_byte)
        txt_file.write(f'{key}.npz {array_shape}\n')  # Write meta information
        if idx % batch == 0:
            txn.commit()
            txn = env.begin(write=True)
    pbar.close()
    txn.commit()
    env.close()
    txt_file.close()
    print('\nFinish writing lmdb.')

if __name__ == '__main__':
    main()
