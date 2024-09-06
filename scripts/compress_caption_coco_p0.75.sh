#!/bin/bash
save_path=output/caption_coco_compression_p0.75
mkdir $save_path

#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 20603 compress_caption_dtp.py --p 0.75 --epoch 5 \
--pretrained pretrained/model_base_caption_capfilt_large.pth \
--config ./configs/caption_coco.yaml \
--output_dir $save_path >$save_path/train.log