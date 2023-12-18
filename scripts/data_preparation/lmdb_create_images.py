# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from basicsr.utils import scandir
from basicsr.utils.create_lmdb import create_lmdb_for_gopro, make_lmdb_from_imgs

opt = {}
# create lmdb for folder containing png images
opt['save_folder'] = '../../datasets/GoPro/test/input_mask'

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
    create_lmdb_for_gopro(opt['save_folder'])

def create_lmdb_for_gopro(folder_path):
    lmdb_path = folder_path + '.lmdb'

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

if __name__ == '__main__':
    main()
