# ------------------------------------------------------------------------
# Copyright (c) 2023 Siwei Li.
# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# Copyright 2023 NAFNet Authors
# ------------------------------------------------------------------------
from os import path as osp
from basicsr.utils.lmdb_util import make_lmdb_from_imgs

from basicsr.utils import scandir

data_dir = '../../datasets/ReLoBlur/'

def prepare_keys(folder_path, suffix='png'):
    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix=suffix, recursive=False)))
    keys = [img_path.split('.{}'.format(suffix))[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys

def create_lmdb_for_reloblur(data_dir:str, stage: str):
    folder_path = osp.join(data_dir, stage, 'target')
    lmdb_path = osp.join(data_dir, stage, 'target.lmdb')

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = osp.join(data_dir, stage, 'input')
    lmdb_path = osp.join(data_dir, stage, 'input.lmdb')

    img_path_list, keys = prepare_keys(folder_path, 'png')
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def run(stage: str):
    opt = {}
    opt['n_thread'] = 20
    opt['compression_level'] = 3
    opt['stage'] = stage

    create_lmdb_for_reloblur(data_dir, stage)


if __name__ == '__main__':
    run('test')