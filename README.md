## Introduction
**This repository contains source code for DF-CLIP: Adapting Visual-Language Models for Generalizable Deepfake Detection Using Multi-Modal Prompt Tuning.** 

Generalizable Deepfake Detection (GDD) aims to detect forged images across different data distributions. Recently, pre-trained Visual-Language Models (VLMs), such as CLIP, have shown unprecedented generalization capabilities in diverse downstream visual tasks, making it a competitive potential choice for GDD tasks. However, due to the lack of deepfake-specific knowledge and cross-modal interactions, VLMs exhibit limited generalization performance on GDD tasks. To address these issues, we propose a novel multi-modal prompt tuning method, namely DF-CLIP, for GDD tasks. Specifically, to integrate deepfake-specific knowledge into VLMs, DF-CLIP employs Visual Prompt Tuning (VPT) and Text Prompt Tuning (TPT) for text and image encoders, respectively. The VPT and TPT can refine the representation space by learnable visual and text prompts, adapting VLMs to deepfake detection tasks. To build the interactions between modalities, we integrate global image features into text prompts, enhancing the model's contextual learning ability and multi-modal synergy. In addition, to further promote cross-modal mutual understanding and improve the generalization performance, we propose a context-enhanced embedding module to fuse local visual features and text embeddings to generate contextual text embeddings, improving the model's forgery semantic perception capabilities. We conduct experiments on multiple GDD benchmarks. Extensive experimental results demonstrate that our proposed method outperforms existing state-of-the-art methods by a large margin.


## Environments
Create a new conda environment and install required packages.
```
conda create -n DF_CLIP python=3.9
conda activate DF_CLIP
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

**Experiments are conducted on a NVIDIA A100.**


#### MVTec-AD and VisA 
1、Download and prepare the dataset: [F+++](https://github.com/ondyari/FaceForensics), [FFIW](https://github.com/tfzhou/FFIW), [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics), [DFDC-P](https://ai.facebook.com/datasets/dfdc/), and [DFD](https://github.com/ondyari/FaceForensics).

2、Use the code under the ./preprocessing file to extract and align faces. Put the processed data under the folder './datasets'.


## Run Experiments

#### Prepare the pre-trained weights
> 1、 Download the CLIP weights pretrained by OpenAI [ViT-B-16-224](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) to **./pretrained_weight/**

#### Run the code

> bash run.sh

#### Test

> bash test.sh
