#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=7 --master_port 20603 compress_nlvr_dtp.py --evaluate \
--pretrained output/nlvr_nlvr2_compression_p0.5/model_base_nlvr_nlvr2_p0.5_compressed.pth --config ./configs/nlvr.yaml \
--output_dir output/nlvr_nlvr2_compression_p0.5