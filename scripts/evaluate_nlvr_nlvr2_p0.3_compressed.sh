#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 20603 compress_nlvr_dtp.py --evaluate \
--pretrained output/nlvr_nlvr2_compression_p0.3/model_base_nlvr_nlvr2_p0.3_compressed.pth --config ./configs/nlvr.yaml \
--output_dir output/nlvr_nlvr2_compression_p0.3