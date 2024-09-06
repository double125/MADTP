#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 20603 compress_retrieval_flickr_dtp.py --evaluate \
--pretrained output/retrieval_flickr_compression_p0.5/model_base_retrieval_flickr_p0.5_compressed.pth --config ./configs/retrieval_flickr.yaml \
--output_dir output/retrieval_flickr_compression_p0.5
