#!/bin/bash
save_path=output/vqa_vqa2_compression_p0.75
mkdir $save_path

#!/bin/bash
python -m torch.distributed.run --nproc_per_node=8 --master_port 20603 compress_vqa_dtp.py --p 0.75 --epoch 3 \
--pretrained pretrained/model_base_vqa_capfilt_large.pth \
--config ./configs/vqa.yaml \
--output_dir $save_path >$save_path/train.log
