#!/bin/bash
save_path=output/retrieval_coco_clip_compression_p0.75
mkdir $save_path

python -m torch.distributed.run --nproc_per_node=8 --master_port 20603 compress_retrieval_clip_dtp.py --p 0.75 --epoch 5 \
--pretrained pretrained/clip_large_retrieval_coco.pth \
--config ./configs/retrieval_coco_clip.yaml \
--output_dir $save_path >$save_path/train.log