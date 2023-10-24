import cv2
import numpy as np
import os
import sys
from multiprocessing import Pool
from os import path as osp
from tqdm import tqdm

from basicsr.utils import scandir
from basicsr.utils.create_lmdb import create_lmdb_for_reloblur

def run(stage: str):
    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3
    opt['stage'] = stage

    opt['input_folder'] = f'./datasets/ReLoBlur/{stage}'
    opt['save_input_folder'] = f'./datasets/ReLoBlur/{stage}/input'
    opt['save_target_folder'] = f'./datasets/ReLoBlur/{stage}/target'

    create_lmdb_for_reloblur(stage)


def extract_images(opt):
    """Extract images.

    Args:
        opt (dict): Configuration dict. It contains:
            input_folder (str): Path to the input folder.
            save_folder (str): Path to save folder.
            n_thread (int): Thread number.
    """
    input_folder = opt['input_folder']
    save_input_folder = opt['save_input_folder']
    if not osp.exists(save_input_folder):
        os.makedirs(save_input_folder)
        print(f'mkdir {save_input_folder} ...')
    else:
        print(f'Folder {save_input_folder} already exists. Exit.')
        sys.exit(1)
    save_target_folder = opt['save_target_folder']
    if not osp.exists(save_target_folder):
        os.makedirs(save_target_folder)
        print(f'mkdir {save_target_folder} ...')
    else:
        print(f'Folder {save_target_folder} already exists. Exit.')
        sys.exit(1)

    #save all image paths in input_folder to img_list, including subdirs
    img_list = []
    for dirpath, _, filenames in os.walk(input_folder):
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img_list.append(os.path.join(dirpath, filename))
    
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(
            worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')

def worker(path, opt):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img.ndim < 2 or img.ndim > 3:
        raise ValueError(f'Image ndim should be 2 or 3, but got {img.ndim}')
    
    folders = path.split(os.sep)
    img_name = '_'.join(folders[-3:-1]) + os.path.splitext(folders[-1])[0] + '.png'
    if "blur" in folders[-1]:
        save_path = os.path.join(opt['save_input_folder'], img_name.replace('_blur', ''))
    else:
        save_path = os.path.join(opt['save_target_folder'], img_name.replace('_sharp', ''))

    cv2.imwrite(
            save_path, img,
            [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])

    process_info = f'Processing {img_name} ...'
    return process_info

if __name__ == '__main__':
    run('test')
    run('train')
