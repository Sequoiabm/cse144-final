# CSE 144 Final Project Report

Transfer Learning Challenge  
Group Members: _TBD_

## 1. Introduction

This project trains a 100-class image classifier for the UCSC CSE 144 Spring 2026
Kaggle transfer-learning challenge. The dataset is small and imbalanced, so the
core approach is to fine-tune strong pretrained image-classification backbones and
ensemble their predictions.

Main result: _TBD after final submission._

## 2. Dataset

The training set contains 1,079 images across 100 class folders named `0` through
`99`. The folder name is the class label, so the implementation uses an explicit
`label = int(folder_name)` mapping and asserts that labels cover exactly `0..99`.
This avoids alphabetical folder-order errors such as `0, 1, 10, 11, ...`.

The test directory contains 1,036 images named numerically from `0.jpg` through
`1035.jpg`. The submission generator predicts every image in the test directory,
not only the 1,000 rows shown in `sample_submission.csv`.

Preprocessing uses RGB conversion, per-backbone timm evaluation transforms, and
ImageNet-style normalization from the pretrained model's resolved data config.

## 3. Implementation

### 3.1 Model

The baseline model is `convnext_small.fb_in22k_ft_in1k` with a new 100-way
classification head and dropout. The next approved high-accuracy gate is
`convnext_base.fb_in22k_ft_in1k`, followed by EffV2-S and ViT-B only if the
ConvNeXt-S + ConvNeXt-B OOF ensemble clears roughly 0.80 OOF accuracy.

All models are fine-tuned from pretrained image-classification weights using only
images in `Data/train/`.

### 3.2 Training

Training uses stratified 5-fold cross-validation, AdamW, layer-wise learning-rate
decay, warmup plus cosine scheduling, gradient clipping, label smoothing,
MixUp/CutMix, and EMA checkpoint selection. CUDA AMP is used when available.

Current canonical settings:

| Setting | Value |
|---|---:|
| Seed | 1337 |
| Folds | 5 stratified folds |
| Epochs | 40 |
| Peak LR | 1e-3 |
| Weight decay | 0.05 |
| Layer decay | 0.8 |
| Dropout | 0.2 |
| Drop path | 0.1 |
| EMA decay | 0.9998 |

## 4. Experiments

| Experiment | Backbone(s) | OOF accuracy | Kaggle accuracy | Notes |
|---|---|---:|---:|---|
| Baseline | ConvNeXt-S | 0.7720 | 0.8000 | Single-backbone result reported from prior run |
| Resolution ablation | ConvNeXt-S 288 / 15 epochs | 0.7600 | _TBD_ | No structural gain from resolution |
| Diversity gate | ConvNeXt-S + ConvNeXt-B | _TBD_ | _TBD_ | Continue only if ~0.80+ OOF |
| Softer augmentation | ConvNeXt-B soft aug | _TBD_ | _N/A_ | OOF-only ablation |
| EffV2-S ensemble | + EffV2-S | _TBD_ | _TBD_ | Run only if diversity gate passes |
| ViT-B ensemble | + ViT-B | _TBD_ | _TBD_ | Run only if EffV2-S helps |

## 5. Results

Final OOF accuracy: _TBD_  
Final Kaggle score: _TBD_  
Leaderboard screenshot: _TBD_

## 6. Discussion

The current evidence suggests that more epochs and a simple 224-to-288 resolution
increase are not sufficient to move beyond the ConvNeXt-S plateau. The next
highest-yield lever is architectural diversity from a stronger ConvNeXt-B model
and later non-ConvNeXt backbones if the OOF ensemble improves.

## 7. Reproducibility

All random seeds are fixed through `seed_everything`. Hyperparameters live in
`config.yaml`, and each training run saves checkpoints, OOF logits, and resolved
configuration snapshots.

Canonical commands:

```bash
python src/train.py --backbone convnext_base
python src/oof_ensemble.py --backbones convnext_small,convnext_base
python src/train.py --backbone convnext_base --aug-profile softer --out checkpoints_soft_aug
python src/predict.py --out submission.csv
```

## 8. Team Contributions

_TBD_

## 9. References

- PyTorch and TorchVision pretrained model documentation.
- timm pretrained model library and model cards.
- UCSC CSE 144 final project directions.
