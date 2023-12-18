### Pretrained NAFNet Models
---

#### Training Details
All models are trained using the Adam optimizer with β1 = 0.9, β2 = 0.9, and a weight decay of 0.001.  
Initial learning rate is 0.001 and gradually reduced to 1e−6 using cosine annealing.  
We adopt a training patch size of 256 × 256, a batch size of 32, and train for a total of 200k iterations.  
Data augmentation techniques such as shuffling, rotation, and flipping are applied during training.

#### Architecture
```
NAFNet-width32
enc_blk_nums: [1, 1, 1, 28]
middle_blk_num: 1
dec_blk_nums: [1, 1, 1, 1]
```

#### Pretrained Models
- **NAFNet-GoPro.pth**: baseline
- **MaskNAFNet-GoPro.pth**: concat
- **SegNAFNet-no-dropout-GoPro.pth**: MAP(ours without dropout)
- **SegNAFNet-GoPro.pth**: ours