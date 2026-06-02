# CSE 144 Final Project — Overview

**Transfer Learning Challenge** · UC Santa Cruz · Spring 2026
**Due:** June 10, 2026, 11:59 PM
**Kaggle:** https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project

---

## 1. The Task

Train an image classifier for a **100-class** dataset using **transfer learning**. Unlike the
earlier CIFAR-10 assignment, this dataset is harder and small, so you are **required to start
from strong pretrained weights** (see [TorchVision models](https://pytorch.org/vision/stable/models.html))
and fine-tune them. The goal is high accuracy on a held-out test set, submitted to Kaggle.

---

## 2. The Data

Located in `Data/`:

```
Data/
├── train/                 # 100 class folders, named "0" .. "99"
│   ├── 0/  -> 0.jpg, 1.jpg, ...
│   ├── 1/  -> ...
│   └── 99/
├── test/                  # unlabeled images: 0.jpg, 1.jpg, ...
└── sample_submission.csv  # submission template (ID, Label)
```

### What the data actually looks like (verified)

- **Classes:** 100 (folders `0`–`99`). Each folder name **is** the label.
- **Images:** colored (RGB), **varying shapes** → resize everything to a fixed size
  (e.g. `224×224`) before training.
- **Train set:** **1,079 total images** across the 100 classes.
- **Test set:** image files are named `0.jpg`–`1035.jpg` (1,036 files on disk).
- **Submission:** `sample_submission.csv` has **1,000 rows** (`0.jpg`–`999.jpg`).

> ⚠️ **Discrepancies vs. the instructions** (worth knowing before you code):
> - The handout says "~10 train images per class" and "1,000 test images (0–999)."
> - In reality the train counts are **uneven**: only 33 classes have exactly 10 images;
>   counts range from **4 to 41** images per class (e.g. class `49` has 41, class `43` has 31).
>   → Expect **class imbalance**; consider weighted sampling / augmentation.
> - The test folder contains **1,036 images** (`0`–`1035`), but the submission template only
>   lists **`0.jpg`–`999.jpg`**. → **Only predict the 1,000 IDs in `sample_submission.csv`.**
>   The extra 36 files (`1000.jpg`–`1035.jpg`) are not scored / not in the template.

### Per-class training image distribution

| Images per class | # of classes |
|------------------|--------------|
| 10               | 33           |
| 11               | 28           |
| 8                | 7            |
| 12               | 5            |
| 14               | 4            |
| 6                | 4            |
| 5                | 4            |
| 13               | 3            |
| 7                | 3            |
| 20               | 2            |
| 9                | 2            |
| 41 / 31 / 16 / 15 / 4 | 1 each  |

### Submission format

A `submission.csv` with two columns, leaving `ID` unchanged and filling in `Label`:

```csv
ID,Label
0.jpg,53
1.jpg,43
2.jpg,48
...
999.jpg,...
```

> 🔑 **Critical label mapping:** Class `"0"` folder → Label `0`, class `"1"` → Label `1`, …,
> class `"99"` → Label `99`. If `ImageFolder`'s alphabetical ordering scrambles this
> (e.g. `"0","1","10","11",...`), your score will be **no better than random**. Build an explicit
> folder-name → integer-label map.

---

## 3. Rules & Constraints

- **Training data:** you may **only** train on images in `train/`.
- **Pretrained weights are required** — fine-tune a strong backbone (ResNet / EfficientNet / ViT, etc.).
- **Beware overfitting:** only ~1,000 training samples, so very large models may overfit.
  Sanity-check GPU memory and training time on a small subset first.
- **Reproducibility required:** fixed random seed, report averaged metrics over multiple runs,
  and provide exact run instructions. If results can't be reproduced, your score may be affected.

---

## 4. Deliverables

1. **Kaggle submission** (`submission.csv`) — public leaderboard is only ~10% of the test set;
   use it to verify format, not to gauge final performance.
2. **Public GitHub repo** containing:
   - All source code.
   - A **README** documenting how to run training **and** inference, plus a **screenshot of your
     Kaggle leaderboard position** (referenced in the README).
   - A **Google Drive link** to trained model weights.
   - A **project report (PDF)** — follow the sample report structure (see below).
3. **Presentation.**
4. Submit the GitHub repo link to **Canvas**.

---

## 5. Grading (max 100)

`S_total = min(100, S_kaggle + S_pres + S_repo)`

| Component | Scoring |
|-----------|---------|
| **Kaggle accuracy** (`S_kaggle`) | Baseline = **60%**. Let `x` = your accuracy (%). Score = `70 + max(0, x − 60)` if you submit, else `0`. Bonus above 60% has **no upper limit**. |
| **Presentation** (`S_pres`) | `p ∈ [0, 10]` if you present; **`−10`** if you don't. |
| **Report + code + weights** (`S_repo`) | `r ∈ [0, 10]` if all submitted; **`−10`** otherwise. |

**Takeaway:** Just clearing the 60% baseline + presenting + submitting the repo gets you to a
strong score; every point above 60% accuracy adds directly on top.

---

## 6. Report Structure (from the sample report)

The PDF report should cover:

1. **Introduction** — problem/goal, why transfer learning fits, summary of approach + main result.
2. **Dataset** — class count & sizes, directory structure & **label mapping** (verify 0–99 match folders), preprocessing (resize, normalization) & augmentation.
3. **Implementation** — pretrained backbone & why, architecture changes (head, pooling, dropout), fine-tuning strategy (frozen layers, unfreeze schedule); training: loss, optimizer, LR/scheduler/batch/epochs/weight decay, hardware/software env.
4. **Experiments** — baseline, hyperparameter tuning + validation method, ablations.
5. **Results** — train/val accuracy & loss curves, Kaggle score, error analysis.
6. **Discussion** — what worked, failure/overfitting cases, limitations & next steps.
7. **Reproducibility** — seeds/determinism, package versions, exact train + submission commands.
8. **Team Contributions** (max 3 members) and **References**.

---

## 7. Suggested Next Steps

1. Build a `Dataset`/`DataLoader` with an explicit folder→label map and a small held-out
   validation split (stratified, given the imbalance).
2. Resize to `224×224`, normalize with ImageNet stats, add augmentation (flip, crop, color jitter).
3. Fine-tune a pretrained backbone (e.g. `resnet50` / `efficientnet_v2` / ViT) with a new 100-way head.
4. Set a fixed seed; track train/val accuracy; iterate to beat the 60% baseline.
5. Predict the **1,000 IDs in `sample_submission.csv`**, write `submission.csv`, submit to Kaggle.
