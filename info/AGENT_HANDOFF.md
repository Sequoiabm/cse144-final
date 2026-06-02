# Agent Handoff — CSE 144 Final Project (Max-Accuracy Transfer Learning)

**Read this first. It is the single entry point for the next agent.** It is self-contained: the
plan, locked decisions, reasoning, file layout, correctness traps, and exact first steps are all
below. You are starting a fresh session (skip-permissions). Begin building the pipeline.

---

## 0. Read these, in order, before writing code
1. **This file** (`info/AGENT_HANDOFF.md`).
2. `info/changelog.md` — living dev log; the "Pipeline plan + design decisions (2026-06-01)" entry
   mirrors this handoff. **You must keep this updated** (changelog protocol below).
3. `info/PROJECT_OVERVIEW.md` — condensed task/data/grading cheat sheet.
4. `info/Models_DeepResearch_Report.md` — backbone research (note its benchmark numbers use
   placeholder citations — do not cite blindly; verify before they enter the report).

---

## 1. The task (what we're building)
Fine-tune pretrained backbones into a **100-class image classifier**; submit `submission.csv` to
Kaggle. Data is tiny and imbalanced: **1,079 train images, 100 classes, 4–41 images/class**
(avg ~11). Test folder has **1,036 files** but we predict only the **1,000 IDs in
`sample_submission.csv`** (`0.jpg`–`999.jpg`). Output columns: `ID,Label` (Label = int 0–99).

**Goal: MAXIMIZE test accuracy** (not just clear 60% — bonus above 60% is uncapped). Class-leading
score is **96%**; explicit target is **≥96%**. The 4-image tail classes are the data-imposed
ceiling — the whole stack aims to squeeze the tail.

### Two non-negotiable correctness traps
1. **Label mapping.** The folder name *is* the label. Use `label = int(folder_name)` directly so the
   100 output logits map 1:1 to labels 0–99. **Never** rely on `ImageFolder`'s alphabetical order
   (`"0","1","10","11",…`) — it scrambles labels and yields ~random accuracy. Assert
   `"0"→0 … "99"→99` at import; this is the first build gate.
2. **Training-data restriction.** Train **only** on `Data/train/`. **No** pseudo-labeling the test
   set, **no** external data. Against the rules; breaks reproducibility.

---

## 2. Locked decisions (do not relitigate)

| Area | Decision | Why |
|------|----------|-----|
| **Compute** | **Kaggle Notebooks (CUDA) = reproducible source of truth.** Code device-agnostic (`cuda → mps → cpu`). Local M2 Pro (MPS) for scratch dev only. | timm 21k backbones (EVA/ViT) are flaky on MPS; CUDA wins on timm support, reproducibility, and zero data-upload friction (Kaggle mounts the comp data). MPS is never the reported env. |
| **Format** | Modular `src/*.py` + `config.yaml` + thin `model.ipynb` Kaggle wrapper + `README.md` + pinned `requirements.txt`. | "Exact train/inference commands" map to CLIs. k-fold × multi-arch × TTA doesn't fit one notebook; `.py` diffs/merges cleanly for a team. Notebook is the 1-click Kaggle entry point. |
| **Ambition** | Build **incrementally**, each layer **gated on a measured OOF/val lift**, but ship the **full stack**. | Incremental = debuggable + ready-made ablation table for the report. Full stack = the committed final. |
| **Checkpoints** | Save per-fold ckpts (raw + EMA) + OOF preds + resolved config frequently. Push final weights to Google Drive (required deliverable). | Kaggle session timeouts; reproducibility; Drive link is graded. |

---

## 3. Target repo structure (create these)
```
src/
  data.py        # folder→label map (+assert), corrupt/format cleaning, stratified k-fold, timm transforms
  model.py       # timm/torchvision backbone factory + 100-way head, EMA wrapper
  train.py       # CLI: fine-tune ONE backbone across folds; saves per-fold ckpt (raw+EMA) + OOF preds
  predict.py     # CLI: load all ckpts, TTA + multi-fold/multi-arch softmax ensemble → submission.csv
  utils.py       # seeding/determinism, device select, metrics, logging, submission validator
config.yaml      # backbones, folds, resolution, aug, lr/llrd, epochs, wd, mixup, ema, seeds, SEED
model.ipynb      # thin Kaggle wrapper: pip install, !python src/train.py ..., !python src/predict.py ...
README.md        # exact train+inference commands, Drive weights link, Kaggle screenshot, pinned versions
requirements.txt # pinned: torch, timm, torchvision, scikit-learn, pandas, pyyaml, pillow, numpy
```
Existing: `Data/` (train/test/sample_submission.csv), `Directions/`, `info/` (docs), root
`model.ipynb` (empty stub — overwrite as the thin wrapper).

---

## 4. Pipeline design (end to end)

### 4.1 Data prep / cleaning (`data.py`)
- `label = int(folder_name)`; **assert** the map covers exactly `{0..99}` and `"0"→0 … "99"→99`.
- Open every train + test image once: convert `L`/`RGBA`/`P` → `RGB`; skip/flag unreadable; log
  anomalies. Confirm test = `0.jpg–1035.jpg`; **predict only the 1,000 sample IDs**.
- Build `(path, label)` list explicitly — never depend on directory iteration order.

### 4.2 Validation — stratified 5-fold (`data.py`)
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)`. Classes with 4 images trigger a
  sklearn *warning* (not error) and land in val for 4/5 folds — acceptable, log it.
- Collect **out-of-fold (OOF) predictions** → leak-free CV accuracy for the whole dataset. OOF is
  the primary offline metric and the **only** thing used to tune ensemble weights/temperature.
  **Never tune on the test set / Kaggle LB.**
- At inference, ensemble **all folds** (every fold's model votes on every test image).

### 4.3 Augmentation (`data.py`, via timm)
- Per-model transform via `timm.data.resolve_data_config` + `create_transform` (correct mean/std/crop
  per checkpoint). Train: RandomResizedCrop, hflip, RandAugment (`rand-m9-mstd0.5-inc1`),
  RandomErasing. Eval: resize→center-crop at the model's native size.
- **MixUp + CutMix** via `timm.data.Mixup` (switch prob ~0.5, label smoothing 0.1) → consumed by
  `SoftTargetCrossEntropy`. Strong aug is the main overfitting defense at ~11 imgs/class.

### 4.4 Backbones — diverse ImageNet-21k (`model.py`)
Three families for ensemble diversity:
- `convnext_small.fb_in22k_ft_in1k` (primary; use the `_384` variant for the high-res pass)
- `tf_efficientnetv2_s.in21k_ft_in1k` (different conv family)
- `eva02_small_patch14_224.mim_in22k_ft_in1k` **or** `vit_base_patch16_224.augreg_in21k_ft_in1k` (transformer)
- Resolution: start 224; for the high-res pass prefer checkpoints *matched* to the resolution
  (`*_384`) over naively upscaling 224 weights.
- New 100-way head with dropout.
- **Open rule check:** confirm the Kaggle competition page imposes no ImageNet-1k-only restriction
  before relying on 21k weights (the project overview does not).

### 4.5 Training recipe (`train.py`)
- **Full fine-tune** with **layer-wise LR decay** (~0.7–0.9); optional short head-only warmup first.
- **AdamW**, weight decay 0.05, **warmup (3–5 ep) + cosine**, ~30–50 epochs, early-stop on EMA val
  acc, grad clipping, **AMP** on CUDA.
- Discriminative LR (low backbone / higher head).
- **EMA** of weights (`ModelEmaV2`, decay ~0.9998); select & report the EMA model.
- **Label smoothing 0.1** (folded into Mixup soft targets).
- Save per fold: best raw ckpt, best EMA ckpt, val curve, OOF logits, resolved config.

### 4.6 Class-imbalance handling (`train.py`) — ABLATION, not assumed
- Try `WeightedRandomSampler` (sqrt-inverse-freq) **or** class-balanced loss weights; keep whichever
  improves OOF. Mixup interacts with sampling — measure, don't stack blindly.

### 4.7 Inference / TTA / ensemble / seed averaging (`predict.py`)
- **TTA:** original + hflip (+ optional multi-scale), average softmax.
- **Ensemble:** average softmax over all folds × backbones × seeds; optionally weight by per-member
  OOF accuracy and apply OOF-tuned temperature.
- **Seed averaging:** repeat final training with 2–3 seeds (last layer of the stack).
- Write `submission.csv` (`ID,Label`) for exactly the 1,000 sample IDs. **Validate**: 1,000 rows,
  header `ID,Label`, labels ∈ 0–99, IDs match `sample_submission.csv`.

---

## 5. Build order (do these in sequence; gate each on measured OOF/val lift; log delta to changelog)
1. `convnext_small.fb_in22k_ft_in1k` + label map + **single** stratified split → confirm a sane val
   number. **(Gate: proves the label map is correct — catches the scramble bug before anything else.)**
2. + 5-fold CV (OOF) → measure lift.
3. + TTA (hflip / multi-scale) → measure lift.
4. + MixUp/CutMix + EMA + label smoothing → measure lift.
5. + 2nd & 3rd diverse backbones, softmax ensemble → measure lift.
6. + seed averaging → measure lift.

**Final committed pipeline = all six shipped together.**

---

## 6. Reproducibility (graded — required)
- `seed_everything` (python/numpy/torch); `cudnn.deterministic=True`, `benchmark=False` where
  feasible (document residual CUDA/MPS nondeterminism). Fixed `SEED` in `config.yaml`.
- Pinned `requirements.txt`; README documents the exact Kaggle base image + versions and the exact
  train + inference commands. Save per-fold ckpts + OOF + config with the weights; push final
  weights to the Google Drive link.

---

## 7. Verification (how to test end-to-end)
- **Label-map assertion test** — `int(folder)` map covers `{0..99}` and `"0"→0…"99"→99` (fail build if not).
- **Overfit sanity** — train on a tiny subset a few epochs, expect ~100% train acc (proves loop + label wiring).
- **Local MPS smoke run** — one fold, few epochs, tiny res: confirms device-agnostic path, data, loss,
  ckpt save/load, and a valid `submission.csv` before any full Kaggle run.
- **OOF CV accuracy** — primary offline metric / gate for every build-order step.
- **Submission validator** (`utils.py`) — 1,000 rows, header, labels ∈ 0–99, IDs match template.
- **Kaggle public LB** — format sanity only (~10% of test); never tune on it.

---

## 8. Environment facts (verified 2026-06-01)
- Dev machine: **Apple M2 Pro, 32 GB, MPS available**. conda `base`: Python 3.11.5, **torch 2.1.2**
  (MPS works), PIL, sklearn, numpy, pandas present; **timm NOT installed**, torchvision unconfirmed.
  → On Kaggle, install/pin timm + a current torch/torchvision. Local MPS is scratch only.
- Data verified: 100 train folders, **1,079** train images, **1,036** test files, **1,000**
  submission rows.

---

## 9. Changelog protocol (must follow)
`info/changelog.md` feeds the final report. After **every** meaningful change/experiment/finding:
add a dated subsection, log what changed + the **measured val/OOF delta**, and add a row to the log
index and the file-changelog table. Keep it detailed — it becomes the report's Experiments/Ablations
section.

## 10. Out of scope here
Report PDF, presentation, and the actual Drive upload come **after** a ≥96% submission exists. This
handoff covers data → training → inference → submission only.

---

### First action for the next agent
Set up `src/` + `config.yaml`, implement `data.py` with the label map + assertion, and execute
**build-order step 1** (single split, convnext_small) to confirm a sane validation number — the
label-map gate — then proceed down the build order, logging each delta to `info/changelog.md`.
