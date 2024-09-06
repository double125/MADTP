#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 10603 compress_caption_dtp.py --evaluate \
--pretrained output/caption_coco_compression_p0.5/model_base_caption_capfilt_large_coco_p0.5_compressed.pth --config ./configs/caption_coco.yaml \
--output_dir output/caption_coco_compression_p0.5