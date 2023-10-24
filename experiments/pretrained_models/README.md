### Pretrained NAFNet Models
---

NAFNet-GoPro-width32
架构：NAFNet架构 width32
训练：GoPro上训练200k iters 32 batchsize
GoPro指标：# psnr: 32.8542	 # ssim: 0.9604
ref：按照给定参数训练

maskNAFNet-GoPro-width32
架构：NAFNet架构 width32 输入concat分割灰度图
训练：GoPro上训练200k iters 32 batchsize
GoPro指标：# psnr: 32.8763	 # ssim: 0.9608

SegNAFNet-GoPro-width32
架构：NAFNet架构 width32 图片通过3*3conv+1*1conv编码到channel-size为x然后mask池化，并concat到输入，训练时对mask做一个0.2的dropout
训练：GoPro上训练200k iters 32 batchsize
seg2 x=2
GoPro指标：# psnr: 32.8318	 # ssim: 0.9604
