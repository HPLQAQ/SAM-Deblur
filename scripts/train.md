### Distributed / Multi GPU
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/GoPro/SegNAFNet.yml --launcher pytorch
```

*If you try distributed, change batch_size_per_gpu and num_gpu in config yaml file. Batch_size = batch_size_per_gpu \* num_gpu which we set 32.*

### Single GPU
```
python basicsr/train.py -opt options/train/GoPro/SegNAFNet.yml
```