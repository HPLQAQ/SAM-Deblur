# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torch import cat as tensor_cat
from torch import from_numpy as tensor_from_numpy
from torchvision.transforms.functional import normalize
from os import path as osp

from basicsr.utils import FileClient, imfrombytes, scandir
import random
import cv2
import numpy as np

class PairedImageGroupMaskDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageGroupMaskDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.mask_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_mask']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder, self.mask_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt', 'mask']
            self.paths = triple_paths_from_lmdb(
                [self.lq_folder, self.gt_folder, self.mask_folder], ['lq', 'gt', 'mask'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = triple_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder, self.mask_folder], ['lq', 'gt', 'mask'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = triple_paths_from_folder(
                [self.lq_folder, self.gt_folder, self.mask_folder], ['lq', 'gt', 'mask'],
                self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))
        
        from io import BytesIO

        # Load mask image
        mask_path = self.paths[index]['mask_path']  # Modify your path function to include mask_path
        img_bytes = self.file_client.get(mask_path, 'mask')  # Fetch bytes using FileClient

        try:
            # Convert bytes to a file-like object
            bio = BytesIO(img_bytes)

            # Load numpy array from this 'file'
            with np.load(bio) as data:
                img_mask = data['masks']
        except Exception as e:
            raise Exception(f"mask path {mask_path} not working: {e}")


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq, img_mask = padding_with_masks(img_gt, img_lq, img_mask, gt_size)

            # random crop
            img_gt, img_lq, img_mask = paired_random_crop_with_masks(img_gt, img_lq, img_mask, gt_size, scale,
                                                gt_path)
            # flip, rotation
            img_gt, img_lq, img_mask = augment_with_masks(img_gt, img_lq, img_mask, self.opt['use_flip'],
                                     self.opt['use_rot'])

        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)
        img_mask = remove_zero_slices(img_mask)
        img_mask = masks2tensor(img_mask, float32=True)
        # img_lq = tensor_cat([img_lq, img_mask], dim=0)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'masks': img_mask,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

def remove_zero_slices(img_mask):
    # Initialize an empty list to hold non-zero slices
    non_zero_slices = []
    
    # Iterate over each slice along the M dimension
    for mask_slice in img_mask:
        # Check if all values are zero in the H*W*1 slice
        if np.any(mask_slice != 0):
            non_zero_slices.append(mask_slice)
    
    # Stack the non-zero slices along the M dimension
    img_mask_non_zero = np.stack(non_zero_slices, axis=0)
    
    return img_mask_non_zero

def triple_paths_from_lmdb(folders, keys):
    assert len(folders) == 3, 'The len of folders should be 3.'
    assert len(keys) == 3, 'The len of keys should be 3.'
    
    lq_folder, gt_folder, mask_folder = folders
    lq_key, gt_key, mask_key = keys

    if not (lq_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb') and mask_folder.endswith('.lmdb')):
        raise ValueError(
            f'{lq_key} folder, {gt_key} folderand {mask_key} folder should both in lmdb '
            f'formats. But received {lq_key}: {lq_folder}; '
            f'{gt_key}: {gt_folder}'
            f'{mask_key}: {mask_folder}')
    # ensure that the three meta_info files are the same
    with open(osp.join(lq_folder, 'meta_info.txt')) as fin:
        lq_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(mask_folder, 'meta_info.txt')) as fin:
        mask_lmdb_keys = [line.split('.')[0] for line in fin]

    # Ensure all three sets of keys are identical
    if set(lq_lmdb_keys) != set(gt_lmdb_keys) or set(gt_lmdb_keys) != set(mask_lmdb_keys):
        raise ValueError('Keys in lq_folder, gt_folder, and mask_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(lq_lmdb_keys):
            paths.append({
                f'{lq_key}_path': lmdb_key,
                f'{gt_key}_path': lmdb_key,
                f'{mask_key}_path': lmdb_key
            })
        return paths

def triple_paths_from_meta_info_file(folders, keys, meta_info_file, filename_tmpl):
    assert len(folders) == 3, 'The len of folders should be 3.'
    assert len(keys) == 3, 'The len of keys should be 3.'

    lq_folder, gt_folder, mask_folder = folders
    lq_key, gt_key, mask_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        lq_name = f'{filename_tmpl.format(basename)}{ext}'
        mask_name = f'{filename_tmpl.format(basename)}.npz'

        paths.append({
            f'{lq_key}_path': osp.join(lq_folder, lq_name),
            f'{gt_key}_path': osp.join(gt_folder, gt_name),
            f'{mask_key}_path': osp.join(mask_folder, mask_name)
        })
    return paths

def triple_paths_from_folder(folders, keys, filename_tmpl):
    assert len(folders) == 3, 'The len of folders should be 3.'
    assert len(keys) == 3, 'The len of keys should be 3.'

    lq_folder, gt_folder, mask_folder = folders
    lq_key, gt_key, mask_key = keys

    lq_paths = list(scandir(lq_folder))
    gt_paths = list(scandir(gt_folder))
    mask_paths = list(scandir(mask_folder))

    assert len(lq_paths) == len(gt_paths) == len(mask_paths), 'Datasets have different number of images.'

    paths = []
    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        lq_path = lq_paths[idx]
        mask_path = mask_paths[idx]

        paths.append({
            f'{lq_key}_path': osp.join(lq_folder, lq_path),
            f'{gt_key}_path': osp.join(gt_folder, gt_path),
            f'{mask_key}_path': osp.join(mask_folder, mask_path)
        })
    return paths

def padding_with_mask(img_lq, img_gt, img_mask, gt_size):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    
    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt, img_mask

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_mask = cv2.copyMakeBorder(img_mask, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    return img_lq, img_gt, img_mask

def padding_with_masks(img_lq, img_gt, img_mask, gt_size):
    h, w, _ = img_lq.shape
    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt, img_mask
    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_mask_padded = []
    for mask_slice in img_mask:
        mask_slice = mask_slice.squeeze(0)
        mask_slice_padded = cv2.copyMakeBorder(mask_slice, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
        mask_slice_padded = np.expand_dims(mask_slice_padded, axis=0)
        img_mask_padded.append(mask_slice_padded)
    img_mask = np.array(img_mask_padded)
    return img_lq, img_gt, img_mask

def paired_random_crop_with_mask(img_gts, img_lqs, img_masks, gt_patch_size, scale, gt_path):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
    if not isinstance(img_masks, list):
        img_masks = [img_masks]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    h_mask, w_mask = img_masks[0].shape
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_mask or w_gt != w_mask:
        raise ValueError(
            f'Size mismatches. GT ({h_gt}, {w_gt}) is not the same as Mask ({h_mask}, {w_mask}).')
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
            f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). '
                         f'Please remove {gt_path}.')

    # randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # crop mask patch
    img_masks = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_masks
    ]

    # crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]

    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    if len(img_masks) == 1:
        img_masks = img_masks[0]
    return img_gts, img_lqs, img_masks

def paired_random_crop_with_masks(img_gts, img_lqs, img_masks, gt_patch_size, scale, gt_path):
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]
    if not isinstance(img_masks, list):
        img_masks = [img_masks]

    h_lq, w_lq, _ = img_lqs[0].shape
    h_gt, w_gt, _ = img_gts[0].shape
    h_mask, w_mask = img_masks[0].shape[2:]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_mask or w_gt != w_mask:
        raise ValueError(
            f'Size mismatches. GT ({h_gt}, {w_gt}) is not the same as Mask ({h_mask}, {w_mask}).')
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(
            f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). Please remove {gt_path}.')

    # Randomly choose top and left coordinates for lq patch
    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    # Crop lq patch
    img_lqs = [
        v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
        for v in img_lqs
    ]

    # Crop mask patch
    img_masks_cropped = []
    for mask_slice in img_masks:
        mask_slice_cropped = mask_slice[:, :, top:top + lq_patch_size, left:left + lq_patch_size]
        img_masks_cropped.append(mask_slice_cropped)
    
    img_masks = np.concatenate(img_masks_cropped, axis=0)

    # Crop corresponding gt patch
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [
        v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
        for v in img_gts
    ]

    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    if len(img_masks_cropped) == 1:
        img_masks = img_masks_cropped[0]
    
    return img_gts, img_lqs, img_masks

def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False, vflip=False):
    hflip = hflip and random.random() < 0.5
    if vflip or rotation:
        vflip = random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if len(img.shape) == 3:
            if hflip:  # horizontal
                cv2.flip(img, 1, img)
                if img.shape[2] == 6:
                    img = img[:,:,[3,4,5,0,1,2]].copy() # swap left/right
            if vflip:  # vertical
                cv2.flip(img, 0, img)
            if rot90:
                img = img.transpose(1, 0, 2)
        elif len(img.shape) == 2:  # For grayscale images
            if hflip:
                cv2.flip(img, 1, img)
            if vflip:
                cv2.flip(img, 0, img)
            if rot90:
                img = img.transpose(1, 0)
        return img

    def _augment_flow(flow):
        if hflip:  # horizontal
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:  # vertical
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        if len(flows) == 1:
            flows = flows[0]
        return imgs, flows
    else:
        if return_status:
            return imgs, (hflip, vflip, rot90)
        else:
            return imgs

def augment_with_masks(img_gt, img_lq, img_mask, hflip=True, rotation=True, flows=None, return_status=False, vflip=False):
    hflip = hflip and random.random() < 0.5
    if vflip or rotation:
        vflip = random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if len(img.shape) == 3:
            if hflip:  # horizontal
                cv2.flip(img, 1, img)
                if img.shape[2] == 6:
                    img = img[:,:,[3,4,5,0,1,2]].copy() # swap left/right
            if vflip:  # vertical
                cv2.flip(img, 0, img)
            if rot90:
                img = img.transpose(1, 0, 2)
        elif len(img.shape) == 2:  # For grayscale images
            if hflip:
                cv2.flip(img, 1, img)
            if vflip:
                cv2.flip(img, 0, img)
            if rot90:
                img = img.transpose(1, 0)
        return img

    def _augment_mask(mask):
        if hflip:
            mask = np.flip(mask, axis=3)
        if vflip:
            mask = np.flip(mask, axis=2)
        if rot90:
            mask = np.rot90(mask, axes=(2, 3))
        return mask

    img_gt = _augment(img_gt)
    img_lq = _augment(img_lq)
    img_mask = _augment_mask(img_mask)

    if return_status:
        return img_gt, img_lq, img_mask, (hflip, vflip, rot90)
    else:
        return img_gt, img_lq, img_mask

def img2tensor(imgs, bgr2rgb=True, float32=True):
    def _totensor(img, bgr2rgb, float32):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = tensor_from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)
    
# Modify a mask2tensor function based on img2tensor
def masks2tensor(mask, float32=True):
    # Note: Assuming the mask is already in grayscale or single channel, no need for color conversion
    if float32:
        mask = mask.astype(np.float32) / 255.0
    mask = tensor_from_numpy(np.ascontiguousarray(mask)).float()
    return mask