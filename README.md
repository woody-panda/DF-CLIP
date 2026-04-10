## Introduction
**This repository contains source code for DF-CLIP: Adapting Visual-Language Models for Generalizable Deepfake Detection Using Multi-Modal Prompt Tuning.** 



## Environments
Create a new conda environment and install required packages.
```
conda create -n DF_CLIP python=3.9
conda activate DF_CLIP
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

**Experiments are conducted on a NVIDIA A100.**


#### Datasets
1、Download and prepare the dataset: [F+++](https://github.com/ondyari/FaceForensics), [FFIW](https://github.com/tfzhou/FFIW), [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics), [DFDC-P](https://ai.facebook.com/datasets/dfdc/), and [DFD](https://github.com/ondyari/FaceForensics).

2、Use the code under the ./preprocessing file to extract and align faces. Put the processed data under the folder './datasets'.


## Run Experiments

#### Prepare the pre-trained weights
> 1、 Download the CLIP weights pretrained by OpenAI [ViT-B-16-224](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) to **./pretrained_weight/**

#### Run the code

> bash run.sh

#### Test

> bash test.sh
