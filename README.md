# MADTP: Multimodal Alignment-Guided Dynamic Token Pruning for Accelerating Vision-Language Transformer

<p align="center"> <a href="https://arxiv.org/pdf/2403.02991.pdf" target="_blank">[Paper]</a> 
<a href="https://arxiv.org/abs/2403.02991" target="_blank">[ArXiv]</a> 
<a href="https://github.com/double125/MADTP" target="_blank">[Code]</a>

<img src="MADTP.png" width="800">

Official implementation of [MADTP: Multimodal Alignment-Guided Dynamic Token Pruning for Accelerating Vision-Language Transformer](https://arxiv.org/abs/2403.02991). 

### What's New 🥳

* (SEP 6, 2024), we released the ```implementation``` and ```scripts``` of MADTP. (Note that ```checkpoints``` and ```logs``` will come soon.)[[Code]](https://github.com/double125/MADTP") 🚩

* (Feb 27, 2024), MADTP: Multimodal Alignment-Guided Dynamic Token Pruning for Accelerating Vision-Language Transformer was accepted by CVPR 2024. [[Paper]](https://arxiv.org/pdf/2403.02991.pdf) [[ArXiv]](https://arxiv.org/abs/2403.02991). 🎉


### Installation
The code is tested on `Pytorch==1.11.0`, `cuda==11.3.1`, and `python==3.8.13`. The dependencies can be installed by:
```
conda env create -f environment.yml
```

### Supported Tasks, Models, and Datasets
Type |  Supported Tasks | Supported Models  | Supported Datasets |
--- | --- | :---: | :---: 
Multi-modal | [Visual Reasoning](https://github.com/double125/MADTP#visual-reasoning-on-the-nlvr2-dataset) | [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/double125/MADTP#visual-reasoning-on-the-nlvr2-dataset)) | [NLVR2](https://lil.nlp.cornell.edu/nlvr/)
Multi-modal |[Image Caption](https://github.com/double125/MADTP#image-caption-on-the-coco-caption-dataset) | [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/double125/MADTP#image-caption-on-the-coco-caption-dataset)) | [COCO Caption](https://cocodataset.org/#home)
Multi-modal |[Visual Question Answer](https://github.com/double125/MADTP#visual-question-answer-on-the-vqav2-dataset) | [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/double125/MADTP#visual-question-answer-on-the-vqav2-dataset)) | [VQAv2](https://visualqa.org/)
Multi-modal |[Image-Text Retrieval](https://github.com/double125/MADTP#image-text-and-text-image-retrieval-on-the-coco-dataset) | [CLIP](https://github.com/openai/CLIP) ([instructions](https://github.com/double125/MADTP#image-text-and-text-image-retrieval-on-the-coco-dataset-with-clip)), [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/double125/MADTP#image-text-and-text-image-retrieval-on-the-coco-dataset)) | [COCO](https://cocodataset.org/#home), [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)
Multi-modal |[Text-Image Retrieval](https://github.com/double125/MADTP#image-text-and-text-image-retrieval-on-the-coco-dataset) | [CLIP](https://github.com/openai/CLIP) ([instructions](https://github.com/double125/MADTP#image-text-and-text-image-retrieval-on-the-flickr30k-dataset-with-clip)), [BLIP](https://github.com/salesforce/BLIP) ([instructions](https://github.com/double125/MADTP#image-text-and-text-image-retrieval-on-the-flickr30k-dataset)) | [COCO](https://cocodataset.org/#home), [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)

### Visual Reasoning on the NLVR2 Dataset

* Dataset & Annotation

    Download the [NLVR2](https://lil.nlp.cornell.edu/nlvr/) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/nlvr.yaml). Download all-in-one annotations (including annotations for Visual Reasoning, Image Caption, VQA, Image-Text Retrieval, and Text-Image Retrieval tasks) from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/nlvr.yaml). See [here](https://github.com/double125/MADTP#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a compressed model with 0.5 reduce ratio:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_nlvr.py --evaluate \
    --pretrained output/nlvr_nlvr2_compression_p0.5/model_base_nlvr_nlvr2_p0.5_compressed.pth \
    --config ./configs/nlvr.yaml \
    --output_dir output/nlvr_nlvr2_compression_p0.5
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/nlvr.yaml). For example, to conduct a compression at 0.5 reduce ratio on 8 A100 GPUs (80G): 
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_nlvr_dtp.py --p 0.5 --epoch 15 \
    --pretrained pretrained/model_base_nlvr.pth \
    --config ./configs/nlvr.yaml \
    --output_dir output/nlvr_nlvr2_compression_p0.5
    ```

* Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    0.3 | <a href="https://drive.google.com/uc?export=download&id=1pcsvlNRzzoq_q6Kaku_Kkg1MFELGoIxE">Download</a> | [Link](./scripts/compress_nlvr_nlvr2_p0.3.sh) | <a href="https://drive.google.com/file/d/1aqiY86op26ceuWp6SFu1kaScqDnAIl1G/view?usp=drive_link">Download</a> | <a href="https://drive.google.com/file/d/1foe-c6qU97QGEz7kNC9OsGJ8OXk7OmQT/view?usp=drive_link">Download</a> | [Link](./scripts/evaluate_nlvr_nlvr2_p0.3_compressed.sh)
    0.5 | <a href="https://drive.google.com/uc?export=download&id=1pcsvlNRzzoq_q6Kaku_Kkg1MFELGoIxE">Download</a> | [Link](./scripts/compress_nlvr_nlvr2_p0.5.sh) | <a href="https://drive.google.com/file/d/1JyYypUDbZVD00ep5SSnQEc6LnOEL-ODT/view?usp=drive_link">Download</a> | <a href="https://drive.google.com/file/d/1R_TgQKlHv6Y6Fh5_ny4fRKNLAva75Frs/view?usp=drive_link">Download</a> | [Link](./scripts/evaluate_nlvr_nlvr2_p0.5_compressed.sh)
    0.6 | <a href="https://drive.google.com/uc?export=download&id=1pcsvlNRzzoq_q6Kaku_Kkg1MFELGoIxE">Download</a> | [Link](./scripts/compress_nlvr_nlvr2_p0.6.sh)| <a href="https://drive.google.com/file/d/1YB8xJee2R7B5PSjzLEJBjmQkBs5XAfIe/view?usp=drive_link">Download</a> | <a href="https://drive.google.com/file/d/1Sg_agxwV04o13d6XnJLblGby5cedtngT/view?usp=drive_link">Download</a> | [Link](./scripts/evaluate_nlvr_nlvr2_p0.6_compressed.sh)
    0.7 | <a href="https://drive.google.com/uc?export=download&id=1pcsvlNRzzoq_q6Kaku_Kkg1MFELGoIxE">Download</a> | [Link](./scripts/compress_nlvr_nlvr2_p0.7.sh)| <a href="https://drive.google.com/file/d/11DbcbzsCjA7mH5gbJQrtrHapobIz12n-/view?usp=drive_link">Download</a> | <a href="https://drive.google.com/file/d/1qcZf5YOl1aDW8S5OEDsIH6lZN4z2UgI8/view?usp=drive_link">Download</a> | [Link](./scripts/evaluate_nlvr_nlvr2_p0.7_compressed.sh)
    0.8 | <a href="https://drive.google.com/uc?export=download&id=1pcsvlNRzzoq_q6Kaku_Kkg1MFELGoIxE">Download</a> | [Link](./scripts/compress_nlvr_nlvr2_p0.8.sh) | <a href="https://drive.google.com/file/d/16K2WIslVVoAzqmMcwvoBWI4gTfxNc8Rv/view?usp=drive_link">Download</a> | <a href="https://drive.google.com/file/d/1l_isAhyRTr7n8qpzXaa8y6hz2BSyR95Y/view?usp=drive_link">Download</a> | [Link](./scripts/evaluate_nlvr_nlvr2_p0.8_compressed.sh)



### Image Caption on the COCO Caption Dataset

* Dataset & Annotation

    Download the [COCO Caption](https://cocodataset.org/#home) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/caption_coco.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/caption_coco.yaml). See [here](https://github.com/double125/MADTP#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a compressed model with 0.5 reduce ratio:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_caption_dtp.py --evaluate \
    --pretrained output/caption_coco_compression_p0.5/model_base_caption_capfilt_large_coco_p0.5_compressed.pth \
    --config ./configs/caption_coco.yaml \
    --output_dir output/caption_coco_compression_p0.5
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/caption_coco.yaml). For example, to conduct a compression at 0.5 reduce ratio on 8 A100 GPUs (80G): 
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_caption_dtp.py --p 0.5 --epoch 5 \
    --pretrained pretrained/model_base_caption_capfilt_large.pth \
    --config ./configs/caption_coco.yaml \
    --output_dir output/caption_coco_compression_p0.5
    ```

<!-- * Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    0.5 | <a href="https://drive.google.com/uc?export=download&id=1qW_0DpQsDc6u9g3fSfTI4g_VXYsMA5s8">Download</a> | [Link](./scripts/compress_caption_coco_p0.5.sh) | <a href="*****r">Download</a> | <a href="*****">Download</a> | [Link](./scripts/evaluate_caption_coco_p0.5_compressed.sh)
    0.75 | <a href="https://drive.google.com/uc?export=download&id=1qW_0DpQsDc6u9g3fSfTI4g_VXYsMA5s8">Download</a> | [Link](./scripts/compress_caption_coco_p0.75.sh)| <a href="*****">Download</a> | <a href="*****">Download</a> | [Link](./scripts/evaluate_caption_coco_p0.75_compressed.sh) -->
    


### Visual Question Answer on the VQAv2 Dataset

* Dataset & Annotation

    Download the [VQAv2](https://visualqa.org/) dataset and [Visual Genome](https://visualgenome.org/) dataset, unzip them under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/vqa.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/vqa.yaml). See [here](https://github.com/double125/MADTP#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a compressed model with 0.5 reduce ratio: (Note that the scripts will generate answers `vqa_result.json`, which should be submitted to the [official server](https://eval.ai/web/challenges/challenge-page/830/overview) to obtain evaluation results.) 
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_vqa_dtp.py --evaluate \
    --pretrained output/vqa_vqa2_compression_p0.5/model_base_vqa_capfilt_large_vqa2_p0.5_compressed.pth \
    --config ./configs/vqa.yaml \
    --output_dir output/vqa_vqa2_compression_p0.5
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/vqa.yaml). For example, to conduct a compression at 0.5 reduce ratio on 8 A100 GPUs (80G): 
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_vqa_dtp.py --p 0.5 --epoch 3 \
    --pretrained pretrained/model_base_vqa_capfilt_large.pth \
    --config ./configs/vqa.yaml \
    --output_dir output/vqa_vqa2_compression_p0.5
    ```

<!-- * Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    0.5 | <a href="https://drive.google.com/uc?export=download&id=18Ihg2NA_puj3_92uVszqonSusLFgmID-">Download</a> | [Link](./scripts/compress_vqa_vqa2_p0.5.sh) | <a href="*****">Download</a> | <a href="*****">Download</a> | [Link](./scripts/evaluate_vqa_vqa2_p0.5_compressed.sh)
    0.75 | <a href="https://drive.google.com/uc?export=download&id=18Ihg2NA_puj3_92uVszqonSusLFgmID-">Download</a> | [Link](./scripts/compress_vqa_vqa2_p0.75.sh)| <a href="*****">Download</a> | <a href="*****">Download</a> | [Link](./scripts/evaluate_vqa_vqa2_p0.75_compressed.sh) -->
    

### Image-Text and Text-Image Retrieval on the COCO Dataset

* Dataset & Annotation

    Download the [COCO](https://cocodataset.org/#home) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/retrieval_coco.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/retrieval_coco.yaml). See [here](https://github.com/double125/MADTP#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a compressed model with 0.5 reduce ratio:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_dtp.py --evaluate \
    --pretrained output/retrieval_coco_compression_p0.5/model_base_retrieval_coco_p0.5_compressed.pth --config ./configs/retrieval_coco.yaml \
    --output_dir output/retrieval_coco_compression_p0.5
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/retrieval_coco.yaml). For example, to conduct a compression at 0.5 reduce ratio on 8 A100 GPUs (80G):
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_dtp.py --p 0.5 --epoch 5 \
    --pretrained pretrained/model_base_retrieval_coco.pth \
    --config ./configs/retrieval_coco.yaml \
    --output_dir output/retrieval_coco_compression_p0.5
    ```

<!-- * Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    0.5 | <a href="https://drive.google.com/uc?export=download&id=19nxvphpnIH2kbV4unL0MDAM_2zlBnruq">Download</a> | [Link](./scripts/compress_retrieval_coco_p0.5.sh) | <a href="*****">Download</a> | <a href="*****">Download</a> | [Link](./scripts/evaluate_retrieval_coco_p0.5_compressed.sh)
    0.75 | <a href="https://drive.google.com/uc?export=download&id=19nxvphpnIH2kbV4unL0MDAM_2zlBnruq">Download</a> | [Link](./scripts/compress_retrieval_coco_p0.75.sh)| <a href="*****">Download</a> | <a href="*****">Download</a> | [Link](./scripts/evaluate_retrieval_coco_p0.75_compressed.sh) -->
    

### Image-Text and Text-Image Retrieval on the Flickr30K Dataset

* Dataset & Annotation

    Download the [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/retrieval_flickr.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/retrieval_flickr.yaml). See [here](https://github.com/double125/MADTP#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a compressed model with 0.5 reduce ratio:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_flickr.py --evaluate \
    --pretrained output/retrieval_flickr_compression_2x/model_base_retrieval_flickr_2x_compressed.pth \
    --config ./configs/retrieval_flickr.yaml \
    --output_dir output/retrieval_flickr_compression_2x
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/retrieval_flickr.yaml). For example, to conduct a compression at 0.5 reduce ratio on 8 A100 GPUs (80G):
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_flickr_dtp.py --p 0.5 --epoch 10 \
    --pretrained pretrained/model_base_retrieval_flickr.pth \
    --config ./configs/retrieval_flickr.yaml \
    --output_dir output/retrieval_flickr_compression_p0.75
    ```

<!-- * Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    0.5 | <a href="https://drive.google.com/uc?export=download&id=1mrd7unZMFMC77Qb_3DAx7MhpZJv4Ptbw">Download</a> | [Link](./scripts/compress_retrieval_flickr_p0.5.sh) | <a href="*****">Download</a> | <a href="*****">Download</a> | [Link](./scripts/evaluate_retrieval_flickr_p0.5_compressed.sh)
    0.75 | <a href="https://drive.google.com/uc?export=download&id=1mrd7unZMFMC77Qb_3DAx7MhpZJv4Ptbw">Download</a> | [Link](./scripts/compress_retrieval_flickr_p0.75.sh)| <a href="*****">Download</a> | <a href="*****">Download</a> | [Link](./scripts/evaluate_retrieval_flickr_p0.75_compressed.sh) -->


### Image-Text and Text-Image Retrieval on the COCO Dataset with CLIP

* Dataset & Annotation

    Download the [COCO](https://cocodataset.org/#home) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/retrieval_coco_clip.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/retrieval_coco_clip.yaml). See [here](https://github.com/double125/MADTP#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a compressed model with 0.5 reduce ratio:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_clip_dtp.py --evaluate \
    --pretrained output/retrieval_coco_clip_compression_p0.5/clip_large_retrieval_coco_p0.5_compressed.pth \
    --config ./configs/retrieval_coco_clip.yaml \
    --output_dir output/retrieval_coco_clip_compression_p0.5
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/retrieval_coco_clip.yaml). For example, to conduct a compression at 0.5 reduce ratio on 8 A100 GPUs (80G): 
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_clip_dtp.py --p 0.5 --epoch 5 \
    --pretrained pretrained/clip_large_retrieval_coco.pth \
    --config ./configs/retrieval_coco_clip.yaml \
    --output_dir output/retrieval_coco_clip_compression_p0.5
    ```

<!-- * Resources

    Reduction | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    0.5 | <a href="https://drive.google.com/uc?export=download&id=10p1oPdiMUqo0MfPul5hCb_h9mCaNCh6q">Download</a> | [Link](./scripts/compress_retrieval_coco_clip_p0.5.sh) | <a href="*****">Download</a> | <a href="*****">Download</a> | [Link](./scripts/evaluate_retrieval_coco_clip_p0.5_compressed.sh)
    0.75 | <a href="https://drive.google.com/uc?export=download&id=10p1oPdiMUqo0MfPul5hCb_h9mCaNCh6q">Download</a> | [Link](./scripts/compress_retrieval_coco_clip_p0.75.sh)| <a href="*****">Download</a> | <a href="*****">Download</a> | [Link](./scripts/evaluate_retrieval_coco_clip_p0.75_compressed.sh) -->


### Image-Text and Text-Image Retrieval on the Flickr30K Dataset with CLIP

* Dataset & Annotation

    Download the [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/) dataset, unzip it under the `datasets` folder, and accordingly modify the `image_root` in [config](./configs/retrieval_flickr_clip.yaml). Download all-in-one annotations  from [this link](https://drive.google.com/uc?export=download&id=19Vk07K3DbQYa68DipJ4dFNcF0_Br7cmD), unzip it under the `annotation` folder, and accordingly modify the `annotation` in [config](./configs/retrieval_flickr_clip.yaml). See [here](https://github.com/double125/MADTP#expected-folder-structures) for expected folder structres.

* Evaluation
  
    Download compressed checkpoints from the table below, put them under the `output` folder, and accordingly modify the `--pretrained` of the scripts. For example, to evaluate a compressed model with 0.5 reduce ratio:
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_clip_dtp.py --evaluate \
    --pretrained output/retrieval_flickr_clip_compression_p0.5/checkpoint_best.pth \
    --config ./configs/retrieval_flickr_clip.yaml \
    --output_dir output/retrieval_flickr_clip_compression_p0.5
    ```

* Compression
  
    Download the uncompressed model from the table below, put it under the `pretrained` folder, and accordingly modify the `pretrained` in [config](./configs/retrieval_flickr_clip.yaml). For example, to conduct a compression at 0.5 reduce ratio on 8 A100 GPUs (80G): 
    ```bash
    python -m torch.distributed.run --nproc_per_node=8 compress_retrieval_clip_dtp.py --p 0.5 --epoch 10 \
    --pretrained pretrained/clip_large_retrieval_flickr.pth \
    --config ./configs/retrieval_flickr_clip.yaml \
    --output_dir output/retrieval_flickr_clip_compression_p0.5
    ```

<!-- * Resources

    Reduce Ratio | Uncompressed Model | Compression Script | Training Log | Compressed Checkpoint | Evaluation Script
    --- | :---: | :---: | :---: | :---: | :---: 
    0.5 | <a href="https://drive.google.com/uc?export=download&id=1-MZP6xQRnmLZr1_pqUK4TvOA8Ic7XCoI">Download</a> | [Link](./scripts/compress_retrieval_flickr_clip_p0.5.sh) | <a href="*****">Download</a> | <a href="*****">Download</a> | [Link](./scripts/evaluate_retrieval_flickr_clip_p0.5_compressed.sh)
    0.75 | <a href="https://drive.google.com/uc?export=download&id=1-MZP6xQRnmLZr1_pqUK4TvOA8Ic7XCoI">Download</a> | [Link](./scripts/compress_retrieval_flickr_clip_p0.75.sh)| <a href="*****">Download</a> | <a href="*****">Download</a> | [Link](./scripts/evaluate_retrieval_flickr_clip_p0.75_compressed.sh) -->

### Common Issues

#### 1. Evaluation with single GPU
   
* For BLIP and CLIP models, evaluate the 2x compressed BLIP model on the NLVR2 dataset as an example:

    ```bash
    python compress_nlvr_dtp.py --evaluate \
    --pretrained output/nlvr_nlvr2_compression_p0.5/checkpoint_best.pth \
    --config ./configs/nlvr.yaml \
    --output_dir output/nlvr_nlvr2_compression_p0.5
    ```

#### 2. Compress with single GPU
   
* For BLIP and CLIP models, compress the BLIP model to half on the NLVR2 dataset as an example:

    ```bash
    python compress_nlvr_dtp.py --p 0.5 --epoch 15 \
    --pretrained pretrained/model_base_nlvr.pth \
    --config ./configs/nlvr.yaml \
    --output_dir output/nlvr_nlvr2_compression_p0.5
    ```

#### 3. Other issues

You can post them on the [Issues](https://github.com/double125/MADTP/issues) page.


### Expected Folder Structures

```
├── annotation
│   ├── answer_list.json
│   ├── coco_gt
│   │   ├── coco_karpathy_test_gt.json
│   │   └── coco_karpathy_val_gt.json
│   ├── ...
├── clip                                               
├── compress_caption_dtp.py             
├── compress_nlvr_dtp.py                  
├── compress ...    
├── configs                                             
├── data                                        
├── datasets
│   └── vision
│       ├── coco
│       ├── flickr
│       ├── NLVR2     
│       ├── ...                                                                               
├── log                                     
├── models            
├── output                                    
├── pretrained
│   ├── bert-base-uncased
│   ├── clip_large_retrieval_coco.pth
│   ├── clip_large_retrieval_flickr.pth
│   ├── ...       
├──                                                                                
├── transform                                                                           
└── utils.py                                
```

### Acknowledgments
This code is built upon <a href="https://github.com/salesforce/BLIP">BLIP</a>, <a href="https://github.com/openai/CLIP">CLIP</a>, <a href="https://github.com/sdc17/UPop">UPop</a>, and <a href=https://github.com/huggingface/pytorch-image-models/tree/main/timm>timm</a>. We thank the original authors for their open-source work.


### Citation
If you find this work useful, please consider citing the corresponding paper:
```bibtex
@article{cao2024madtp,
  title={MADTP: Multimodal Alignment-Guided Dynamic Token Pruning for Accelerating Vision-Language Transformer},
  author={Jianjian, Cao and Peng, Ye and Shengze, Li and Chong, Yu and Yansong, Tang and Jiwen, Lu and Tao, Chen},
  journal={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

