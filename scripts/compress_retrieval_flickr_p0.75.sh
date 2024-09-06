#!/bin/bash
save_path=output/retrieval_flickr_compression_p0.65
mkdir $save_path

#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 20603 compress_retrieval_flickr_dtp.py --p 0.65 --epoch 10 \
--pretrained pretrained/model_base_retrieval_flickr.pth \
--config ./configs/retrieval_flickr.yaml \
--output_dir $save_path >$save_path/train.log