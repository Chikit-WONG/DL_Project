# Visual_EEG_Decoding

Leveraging Visual Blur Perception Characteristics for EEG Decoding [AAAI 2026]

## Introduction

This is the official implementation for Leveraging Visual Blur Perception Characteristics for EEG Decoding [AAAI 2026].

In this paper, we propose a novel visual decoding framework inspired by human perceptual blurring, achieving a top-1 accuracy of 80% and a top-5 accuracy of 96.9%, surpassing previous state-of-the-art methods by margins of 29.1% and 17.2%, respectively. These findings highlight the potential of incorporating perceptual properties into EEG-based visual decoding.

![1763477641384](image/README/1763477641384.png)

## How to use

1. Data Preparation

Download the Things-image from the [OSF repository](https://osf.io/jum2f/files/osfstorage), Things-EEG from the [OSF repository](https://osf.io/b83fj/overview).  We provided the processed Things-EEG data and the pretrained CLIP model weights on [Quark Netdisk](https://pan.quark.cn/s/3fe3136bfafb). If the processed data is downloaded, the following three processing steps (Data Preparation, EEG Data Process, and Image Data Process) can be skipped.

Arrange the data according to the following directory:

```
data
├── things_eeg
│   ├── Image_set
│   │   ├── train_images
│   │   └── test_images
│   └── Raw_eeg
│       ├── sub-01
│       ├── ...
│       └── sub-10
```

2. EEG Data Process

```bash
# Setting the subject number 'sub' from 1 to 10 to process each subject's EEG data.
python preprocess/process_eeg.py --subject sub
```

3. Image Data Process

```bash
# Making multi-blur clip features.
python preprocess/process_image.py
```

After the above steps, the directory structure is as follows:

```
data
├── things_eeg
│   ├── Image_set
│   │   ├── train_images
│   │   └── test_images
│   ├── Image_feature
│   │   ├──MultiBlur_RN50_train.pt
│   │   └──MultiBlur_RN50_test.pt
│   └── Preprocessed_data
│       ├── sub-01
│       ├── ...
│       └── sub-10
```

6. Run

```bash
python main_eeg.py
```

## Acknowledgement

We extend our gratitude to the prior works [UBP](https://github.com/HaitaoWuTJU/Uncertainty-aware-Blur-Prior/tree/main) and [NICE-EEG](https://github.com/eeyhsong/NICE-EEG) for their pioneering contributions to this field.

## **Future Plans**

More code update in progress...

## Citation

```
@inproceedings{liu2026leveraging,
  title={Leveraging Visual Blur Perception Characteristics for EEG Decoding},
  author={Liu, Wenchao and Li, Hongwei and Xu, Zhouyang and Ma, Lin and Li, Haifeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={21},
  pages={17580--17588},
  year={2026}
}
```
