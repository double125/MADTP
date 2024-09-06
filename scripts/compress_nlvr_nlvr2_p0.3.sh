#!/bin/bash
save_path=output/nlvr_nlvr2_compression_p0.3
mkdir $save_path

python -m torch.distributed.run --nproc_per_node=8 --master_port 20603 compress_nlvr_dtp.py --epoch 15 --p 0.3 \
--pretrained pretrained/model_base_nlvr.pth \
--config ./configs/nlvr.yaml \
--output_dir $save_path >$save_path/train.log
