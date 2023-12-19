## Distributed
```
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/GoPro/SegNAFNet.yml --launcher pytorch
```

## Single GPU
```
python basicsr/test.py -opt options/test/GoPro/SegNAFNet.yml
```