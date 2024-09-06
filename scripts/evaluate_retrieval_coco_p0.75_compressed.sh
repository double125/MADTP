#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 20603 compress_retrieval_dtp.py --evaluate \
--pretrained output/retrieval_coco_compression_p0.75/model_base_retrieval_coco_p0.75_compressed.pth --config ./configs/retrieval_coco.yaml \
--output_dir output/retrieval_coco_compression_p0.75