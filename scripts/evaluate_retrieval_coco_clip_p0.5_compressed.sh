#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 20603 compress_retrieval_clip_dtp.py --evaluate \
--pretrained output/retrieval_coco_clip_compression_p0.5/clip_large_retrieval_coco_p0.5_compressed.pth --config ./configs/retrieval_coco_clip.yaml \
--output_dir output/retrieval_coco_clip_compression_p0.5
