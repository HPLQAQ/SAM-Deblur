# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.seg_local_arch import Seg_Local_Base
from basicsr.models.archs.NAFNet_arch import NAFBlock


class SegNAFNet(nn.Module):

    def __init__(self, img_channel=3, seg_channel=3, width=16, middle_blk_num=1, mask_dropout = 0.5, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()
        
        self.img_channel = img_channel
        self.mask_dropout = mask_dropout

        self.pre_seg = nn.Conv2d(in_channels=img_channel, out_channels=seg_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.pre_seg_in = nn.Conv2d(in_channels=seg_channel, out_channels=seg_channel, kernel_size=1, padding=0, stride=1, groups=1,
                              bias=True)
        self.intro = nn.Conv2d(in_channels=img_channel+seg_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def mask_average(self, image, masks, dropout, is_training=True):
        """
        Compute the average value for each mask area in the image and assign it to the entire mask area.
        
        Parameters:
        - image (torch.Tensor): A tensor of shape (C, H, W) representing an image.
        - masks (torch.Tensor): A tensor of shape (M, 1, H, W) representing M masks.
        
        Returns:
        - result (torch.Tensor): A tensor of shape (C, H, W) with the same shape as the input image,
        containing the average values for each mask area.
        """
        image = image.permute(1, 2, 0)
        # Expand image shape to (1, H, W, C)
        image_exp = image.unsqueeze(0)
        # Expand masks shape to (M, H, W, 1)
        masks_exp = masks.squeeze(1).unsqueeze(-1)
        
        # Dropout some masks
        if is_training:
            M = masks_exp.shape[0]
            drop_count = int(dropout * M)
            drop_indices = torch.randperm(M)[:drop_count]
            masks_exp[drop_indices] = 0
        
        # Compute the area of each mask (sum over H, W dimensions)
        mask_areas = masks_exp.sum(dim=(1, 2), keepdim=True)
        # Compute the sum of pixel values in each mask area (sum over H, W dimensions)
        masked_sums = (image_exp * masks_exp).sum(dim=(1, 2), keepdim=True)
        
        # Compute the average pixel value in each mask area (broadcast over H, W, C dimensions)
        avg_values = masked_sums / (mask_areas + 1e-8)
        
        # Initialize the result tensor with zeros
        result = torch.zeros_like(image)
        # Update the result tensor with the average values (broadcast over H, W dimensions)
        result = (avg_values * masks_exp).sum(dim=0)
        
        # Create a mask for uncoverd area
        remaining_mask = 1 - masks_exp.sum(dim=0, keepdim=True).clamp_(0, 1)
        remaining_area = remaining_mask.sum()
        remaining_sum = (image_exp * remaining_mask).sum(dim=(1, 2), keepdim=True)
        remaining_avg = remaining_sum / (remaining_area + 1e-8)
        result += (remaining_avg * remaining_mask).squeeze(0)
        
        result = result.permute(2, 0, 1)
        return result

    def forward(self, inp, masks):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        seg_info = self.pre_seg(inp)
        seg_info = self.pre_seg_in(seg_info)
        #seg_info = F.relu(seg_info)

        for b in range(B):
            seg_info[b] = self.mask_average(seg_info[b], self.check_mask_size(masks[b]), dropout=self.mask_dropout, is_training=self.training)
        
        x = self.intro(torch.cat([inp, seg_info], dim=1))

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        x = x[:, :self.img_channel, :H, :W] + inp[:, :self.img_channel, :H, :W]

        return x
    
    def check_mask_size(self, mask):
        # Assuming mask has shape [M, 1, H, W]
        M, C, H, W = mask.size()
        mod_pad_h = (self.padder_size - H % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - W % self.padder_size) % self.padder_size
        
        # Pad the mask using constant padding of 0
        mask_padded = F.pad(mask, (0, mod_pad_w, 0, mod_pad_h), mode='constant', value=0)
        
        return mask_padded

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class SegNAFNetLocal(Seg_Local_Base, SegNAFNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Seg_Local_Base.__init__(self)
        SegNAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


if __name__ == '__main__':
    img_channel = 3
    seg_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = SegNAFNet(img_channel=img_channel, seg_channel=seg_channel, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)


    inp_shape = (4, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
