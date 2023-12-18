## Distributed
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/GoPro/SegNAFNet.yml --launcher pytorch

## Single GPU
python basicsr/train.py -opt options/train/GoPro/SegNAFNet.yml