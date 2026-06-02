# CSE 144 Final Project — Changelog & Dev Log

Living reference for experiments, decisions, and report writing.  
**Also see:** `PROJECT_OVERVIEW.md` (condensed cheat sheet).

| Field | Value |
|-------|-------|
| Course | CSE 144 — Applied Machine Learning |
| Project | Transfer Learning Challenge (Final) |
| Due | June 10, 2026, 11:59 PM |
| Kaggle | https://www.kaggle.com/competitions/ucsc-cse-144-spring-2026-final-project |
| Directions | `Directions/md/CSE144_Project_Page.md`, `Directions/md/CSE144_Sample_Report.md` |
| Data root | `Data/` |
| Code | `model.ipynb` (WIP) |

---

## Log index

| Date | Section | Summary |
|------|---------|---------|
| 2026-06-01 | [Initial discovery](#initial-discovery-2026-06-01) | Read handout, explored `Data/`, documented gotchas |
| 2026-06-01 | [Goal change + model research + strategy](#goal-change--model-research--optimization-strategy-2026-06-01) | Pivot to max-accuracy; backbone shortlist, 384 caveat, timm/21k decision, optimization plan |
| 2026-06-01 | [Pipeline plan + design decisions](#pipeline-plan--design-decisions-2026-06-01) | Locked compute/format/ambition; full pipeline design, build order, repro & verification; handoff doc written |
| 2026-06-01 | [Pipeline implemented + step-1 label-map gate PASSED](#pipeline-implemented--build-order-step-1-passed-2026-06-01) | Wrote full `src/` + config + notebook + README + requirements; fixed two real bugs (MPS autocast, EMA cold-start); label-map gate passes (val 0→0.45 on 10 classes) |
| 2026-06-01 | [Kaggle handoff prep + ViT tag confirmed](#kaggle-handoff-prep--vit-tag-confirmed-2026-06-01) | Confirmed exact ViT tag (21k/224); notebook scoped to ConvNeXt-S baseline first; git repo initialized + first commit; baseline OOF to be produced on Kaggle CUDA (push + run need user creds) |

*(Add a row here whenever you make a meaningful change.)*

---

## Initial discovery (2026-06-01)

### What we did

1. Read `Directions/md/CSE144_Project_Page.md` and `CSE144_Sample_Report.md`.
2. Listed and sampled `Data/train`, `Data/test`, and `Data/sample_submission.csv`.
3. Counted images per class and compared counts to the written instructions.
4. Wrote `PROJECT_OVERVIEW.md` as a one-page project summary.

### Task (from handout)

- **Goal:** 100-way image classification using **transfer learning** (pretrained backbone + fine-tune).
- **Why transfer learning:** ~1k labeled train images, mixed sources/shapes, hard vs. CIFAR-10; pretrained weights required.
- **Inference:** Predict labels for test images → `submission.csv` → Kaggle.
- **Training constraint:** May **only** use images under `Data/train/`.

### Dataset layout (expected vs. verified)

**Expected (handout):**

```
Data/
├── train/
│   ├── 0/   … ~10 images
│   ├── 1/
│   └── 99/
├── test/
│   ├── 0.jpg … 999.jpg   (1000 images)
└── sample_submission.csv
```

**Verified on disk:**

| Item | Handout says | Actually on disk |
|------|----------------|------------------|
| Train classes | 100 folders `0`–`99` | ✓ 100 folders |
| Train images total | ~10 per class → ~1000 | **1,079** images |
| Test images | 1000 (`0.jpg`–`999.jpg`) | **1,036** files (`0.jpg`–`1035.jpg`) |
| Submission rows | 1000 | ✓ **1,000** rows in `sample_submission.csv` |

### Data gotchas (must not forget)

#### 1. Label mapping — #1 failure mode

- Folder name **is** the label: `train/0/` → class **0**, …, `train/99/` → class **99**.
- **Do not** rely on `ImageFolder` default class order (alphabetical: `0, 1, 10, 11, …, 2, 20, …`) — that scrambles labels and Kaggle score ≈ random.
- **Action:** Build explicit `folder_name → int(label)` map; use same map at inference.

#### 2. Class imbalance (handout understates this)

- Handout: “~10 images per class.”
- Reality: **4–41 images per class**; only **33/100** classes have exactly 10.
- Notable outliers: class **49** → 41 images; class **43** → 31; several with 20, 16, 15.
- **Action:** Stratified val split; consider weighted loss / sampler; heavier aug on rare classes.

**Per-class count distribution (how many classes have N images):**

| Images/class | # classes |
|--------------|-----------|
| 10 | 33 |
| 11 | 28 |
| 8 | 7 |
| 12 | 5 |
| 14 | 4 |
| 6 | 4 |
| 5 | 4 |
| 13 | 3 |
| 7 | 3 |
| 20 | 2 |
| 9 | 2 |
| 41, 31, 16, 15, 4 | 1 each |

#### 3. Test folder vs. submission template mismatch

- `Data/test/` has **1,036** `.jpg` files.
- `sample_submission.csv` lists only **`0.jpg`–`999.jpg`** (1000 IDs).
- Extra files **not in template:** `1000.jpg` … `1035.jpg` (36 files).
- **Action:** Generate predictions **only** for IDs in `sample_submission.csv`; leave `ID` column unchanged.

#### 4. Variable image sizes

- All images are RGB but **shapes differ**.
- **Action:** Resize consistently (e.g. 224×224) before training; document preprocessing in report.

#### 5. Small data + large models

- ~1k train samples → large pretrained nets may **overfit**.
- **Action:** Try subset first for GPU memory / runtime; regularization, aug, early stopping, maybe smaller backbone.

### Submission format

```csv
ID,Label
0.jpg,<predicted_class_0-99>
1.jpg,...
...
999.jpg,...
```

- Columns: `ID`, `Label` (integer 0–99).
- `sample_submission.csv` labels are **dummy** (e.g. 53, 43) — replace with model predictions.

### Requirements checklist (grading & deliverables)

**Kaggle**

- [ ] `submission.csv` uploaded
- [ ] Pass **60%** baseline (score formula: `70 + max(0, accuracy% − 60)`)
- [ ] Public leaderboard ≈ 10% of test — use for **format check only**, not final performance

**Canvas / GitHub**

- [ ] Public repo: all source code
- [ ] README: how to train + run inference
- [ ] README: Kaggle leaderboard **screenshot** (team position)
- [ ] PDF report (see report outline below)
- [ ] Google Drive link to trained weights (in repo)

**Other**

- [ ] Presentation (else **−10** on `S_pres`)
- [ ] Report + code + weights (else **−10** on `S_repo`)

**Total score:** `S_total = min(100, S_kaggle + S_pres + S_repo)`

### Reproducibility (required by handout)

- Fixed random seed(s)
- Report metrics averaged over multiple runs where applicable
- Exact commands to train and produce `submission.csv`
- Package versions / environment documented  
→ Non-reproducible results may hurt competition score.

### Report outline (from sample report — fill as we go)

| Section | Status | Notes |
|---------|--------|-------|
| 1. Introduction | ⬜ | Goal, why transfer learning, summary result |
| 2. Dataset | 🟡 | Use gotchas + counts above; label mapping |
| 3. Implementation (model + training) | ⬜ | Backbone, head, fine-tune strategy, hyperparams |
| 4. Experiments | ⬜ | Baseline, tuning, ablations |
| 5. Results | ⬜ | Val metrics, Kaggle score, optional error analysis |
| 6. Discussion | ⬜ | What worked, failures, next steps |
| 7. Reproducibility | ⬜ | Seeds, versions, commands |
| 8. Team contributions | ⬜ | Max 3 members |
| 9. References | ⬜ | TorchVision, papers, etc. |

---

## Goal change + model research + optimization strategy (2026-06-01)

### 0. Goal change — clear-60% → maximize accuracy

- **Previous framing:** beat the 60% Kaggle baseline.
- **New framing:** **maximize held-out test accuracy.** Bonus above 60% is **uncapped**
  (`S_kaggle = 70 + max(0, x − 60)`), so every legitimate accuracy point is worth pursuing.
- **Honest ceiling caveat:** with ~11 images/class on average and some classes at **4–8 images**,
  the tail classes impose a hard data ceiling. The realistic aim is to get **as close to that
  data-imposed ceiling as possible**, not to chase arbitrarily high accuracy.

### 0a. Rules re-confirmed (correctness traps)

- **Label mapping:** explicit folder-name → int map, verify `"0"→0 … "99"→99`. Never trust
  `ImageFolder` alphabetical ordering. (See [Data gotchas](#1-label-mapping--1-failure-mode).)
- **Training data restriction:** train **only** on images in `Data/train/`. **No pseudo-labeling
  the test set** — against the rules and breaks reproducibility.

### 1. Model research findings (deep-research pass)

Full report: `info/Models_DeepResearch_Report.md`. Summary of conclusions:

- Ranking was by **small-data transfer fit** (feature quality → overfit risk → 224 fit →
  library maturity), **NOT raw ImageNet top-1**. The research cites a finding that higher ImageNet
  accuracy does **not** reliably predict better fine-tuned accuracy, and that **CNNs beat
  Swin-family transformers** for resource-efficient low-data fine-tuning.
- Closest published analogues to our regime: **Flowers-102** (2,040 imgs / 102 classes) and the
  **1,000-image Tiny ImageNet** low-data setting.

**Within-torchvision candidate shortlist (ranked):**

| Rank | Model | Weights enum | Native input | Note |
|------|-------|-------------|--------------|------|
| 1 | `efficientnet_v2_s` | `EfficientNet_V2_S_Weights.IMAGENET1K_V1` | **384** | Best low-data transfer evidence; **least clean 224 fit** |
| 2 | `convnext_tiny` | `ConvNeXt_Tiny_Weights.IMAGENET1K_V1` | 224 | Strongest **224-native** choice; best natural-image transfer overall |
| 3 | `regnet_y_3_2gf` | `RegNet_Y_3_2GF_Weights.IMAGENET1K_V2` | 224 | Balanced: smaller, 224-native, strong transfer |
| hedge | `efficientnet_b1` | `EfficientNet_B1_Weights.IMAGENET1K_V2` | 240 | Small-model fallback when overfitting dominates |

### 2. The 384-vs-224 caveat

- **EfficientNet_V2_S's official torchvision weights were validated at a 384 crop.** It is the
  **least clean fit** for a 224 pipeline.
- **Action:** either run EfficientNet_V2_S at **384**, or prefer the **224-native**
  `convnext_tiny` / `regnet_y_3_2gf` for a clean 224 pipeline.

### 3. Decision — bring in ImageNet-21k pretrained timm backbones

- The deep-research pass deliberately stayed **inside torchvision**. For the max-accuracy goal we
  will **also** pull in **ImageNet-21k pretrained `timm` backbones**, since 21k pretraining gives
  the **biggest gains in low-data settings**. Candidates:
  - `convnext_tiny.fb_in22k_ft_in1k`
  - `convnext_small.fb_in22k_ft_in1k`
  - `eva02_small_patch14_224`
  - ViT-B/16 in21k (e.g. `vit_base_patch16_224.augreg_in21k_ft_in1k`)
- **Rule check:** the project overview imposes **no restriction to ImageNet-1k pretraining**
  (only that we train on `Data/train/` images and start from pretrained weights). 21k-pretrained
  backbones are therefore allowed. *(Re-verify against any Kaggle competition-page rule before
  final submission.)*

### 4. Planned optimization strategy (rough order of impact)

1. **ImageNet-21k pretrained backbones** via `timm`.
2. **Ensemble** multiple diverse architectures (average softmax outputs).
3. **Stratified k-fold CV** (e.g. 5-fold): train per fold, ensemble all folds at inference —
   reliable val estimate + uses all data.
4. **Test-time augmentation** (flips, averaged).
5. **Higher input resolution** where the backbone supports it (256–384).
6. **Heavy regularization/augmentation:** RandAugment, MixUp + CutMix, label smoothing ~0.1,
   stochastic depth, dropout, weight decay.
7. **Discriminative learning rates** (low backbone / higher head), **warmup + cosine** schedule,
   **EMA** of weights.
8. **Class-imbalance handling:** weighted loss or weighted sampler for small-count classes.

### 5. ⚠️ Unverified-benchmark caveat (must resolve before report)

- The deep-research output leaned heavily on a single **"2025 backbone benchmark"** for its
  specific low-data numbers (e.g. the 1,000-image Tiny ImageNet percentages, per-dataset transfer
  scores).
- Its **citation tokens were placeholders, not real links** (`citeturn…`).
- **Do NOT cite those numbers blindly.** Every figure must be **verified against the real primary
  sources** before it enters the report's References / Experiments sections.

---

## Pipeline plan + design decisions (2026-06-01)

Full build instructions for the next agent live in **`info/AGENT_HANDOFF.md`** (self-contained:
plan, decisions, reasoning, build order, verification). This entry records the plan and the
*reasoning* behind each locked decision for the report.

### Target & success criterion
- **Maximize test accuracy.** Class-leading Kaggle score is **96%** → explicit target **≥96%**.
- Realistic ceiling: 4-image tail classes cap us; the stack is aimed at squeezing the tail.

### Locked decisions + reasoning

**1. Compute — Kaggle Notebooks (CUDA) = reproducible source of truth; code device-agnostic
(`cuda → mps → cpu`); local M2 Pro (MPS) for scratch dev only.**
- *Why CUDA over MPS:* the plan needs 21k `timm` backbones (EVA-02, ViT, SwinV2); MPS support for
  those ops is hit-or-miss (silent CPU fallback or errors), and MPS determinism is weaker. A
  Mac-only final environment is a reproducibility liability (graders can't re-run it).
- *Why Kaggle over Colab:* it's a Kaggle competition — data is already mounted (zero upload), submit
  from the same notebook, and the runtime is a fixed version-pinned image (ideal for "pinned
  versions + exact commands"). Dataset is tiny, so we don't need Colab Pro's A100 speed. Colab is a
  fallback only.
- *Keep device-agnostic anyway:* lets us debug quickly on the M2 while graded runs go on Kaggle —
  but local MPS runs are scratch, never reported.

**2. Format — modular `src/*.py` + `config.yaml` + thin `model.ipynb` Kaggle wrapper.**
- *Why:* "exact runnable train + inference commands" maps cleanly to CLIs
  (`python src/train.py --config …`, `python src/predict.py …`). k-fold × multi-arch × ensemble ×
  TTA × ablations does not fit one notebook (2,000-line scroll, untraceable results, painful
  re-runs). Config-driven scripts make each ablation a flag flip → directly feeds the report's
  Experiments section. A public GitHub repo + up-to-3-person team means `.py` diffs/merges cleanly
  where notebooks conflict. The thin `model.ipynb` keeps the 1-click Kaggle run + LB screenshot.

**3. Ambition — build incrementally (each layer gated on a measured OOF/val lift) but ship the
full stack.**
- *Why:* incremental validation catches the label-scramble bug first and yields a ready-made
  ablation table; the full stack is the committed final, not "if time allows."
- **Final committed pipeline = multi-arch 21k ensemble × 5-fold CV × TTA × MixUp/CutMix × EMA ×
  label smoothing × seed averaging.**

### Pipeline design (summary; full detail in handoff)
- **Data:** `label = int(folder_name)` + hard assert `"0"→0…"99"→99`; convert non-RGB → RGB; skip
  corrupt; predict only the 1,000 sample IDs.
- **Validation:** `StratifiedKFold(5, shuffle, seed)`; **OOF predictions** = leak-free CV metric and
  the *only* thing used to tune ensemble weights/temperature (never the test set / LB).
- **Aug:** per-model `timm` transforms (RandAugment, RRC, hflip, RandomErasing) + MixUp/CutMix +
  label smoothing 0.1 (SoftTargetCE).
- **Backbones (diverse 21k):** `convnext_small.fb_in22k_ft_in1k` (+`_384`),
  `tf_efficientnetv2_s.in21k_ft_in1k`, and `eva02_small_patch14_224` / `vit_base_patch16_224.augreg_in21k_ft_in1k`.
- **Training:** full fine-tune + layer-wise LR decay, AdamW, warmup+cosine, EMA (~0.9998), AMP,
  grad clip, discriminative LR, early stop on EMA val acc.
- **Imbalance:** weighted sampler / class-balanced loss as a *measured ablation*, not assumed.
- **Inference:** hflip(+multi-scale) TTA → softmax ensemble over folds × backbones × seeds →
  validated `submission.csv`.

### Build order (validation-gated)
1. convnext_small + label map + single split → sane val (label-map gate).
2. + 5-fold CV → 3. + TTA → 4. + MixUp/CutMix + EMA + label smoothing → 5. + 2nd/3rd backbone
   ensemble → 6. + seed averaging. Log the measured delta of every step here.

### Reproducibility & verification
- Fixed `SEED`, `cudnn.deterministic`, pinned `requirements.txt`, exact commands in README, weights
  → Google Drive. Verify via: label-map assertion, tiny-subset overfit sanity, local MPS smoke run,
  OOF CV accuracy, submission-format validator, and LB used for format only.

### Open items to resolve during build
- Confirm no Kaggle competition-page rule restricts pretraining to ImageNet-1k.
- Verify the deep-research "2025 backbone benchmark" numbers against real sources before citing.
- Confirm 5-fold vs fewer folds given 4-image tail classes (5-fold warns but proceeds).

---

## Pipeline implemented + build-order step 1 PASSED (2026-06-01)

### What was built
Implemented the full modular pipeline exactly per `AGENT_HANDOFF.md`:
- `src/utils.py` — `seed_everything` (+cudnn deterministic), device select (`cuda→mps→cpu`),
  `amp_enabled` (CUDA-only), config load/save, `resolve_backbone` (merges per-backbone overrides),
  metrics, logging, and a hard `validate_submission` (rows, exact ID order, header, labels∈0..99).
- `src/data.py` — `build_label_map` with the **hard assert `"0"→0 … "99"→99`**, explicit
  `(paths,labels)` built from the map (never dir-iteration order), RGB-conversion guard,
  `StratifiedKFold`, per-model timm transforms (RandAug/RRC/hflip/color-jitter/RandomErasing for
  train; resize→center-crop for eval), Train/Test datasets, sqrt-inverse-freq sample weights.
- `src/model.py` — timm backbone factory (drop_rate + drop_path_rate, dynamic `img_size` for
  ViT/EVA), layer-wise-LR-decay param groups (`param_groups_layer_decay`, verified 76 groups
  scaling 0.0003→1.0 on ConvNeXt-S), EMA wrapper.
- `src/train.py` — CLI to fine-tune ONE backbone across folds: AdamW + LLRD (lr_scale baked into
  per-group lr), `CosineLRScheduler` (warmup+cosine), MixUp/CutMix→`SoftTargetCrossEntropy` (or
  CE+label-smoothing when mixup off), grad clip, AMP (CUDA), EMA; early-stop on EMA val acc; saves
  per-fold raw+EMA ckpts, OOF logits (`oof.npz`), resolved config. Quick-run flags
  (`--folds/--epochs/--batch-size/--img-size/--limit-classes/--no-mixup/--device`).
- `src/predict.py` — loads every fold of every `config.ensemble` backbone, hflip TTA, softmax
  ensemble (optional OOF-weighting + temperature), writes a **validated** `submission.csv` for the
  exact 1,000 template IDs.
- `config.yaml` (SEED=1337 + all hyperparams), thin `model.ipynb` Kaggle wrapper, `README.md`
  (exact train/inference commands + repro), pinned `requirements.txt`.
- **Corrected backbone tags vs. the plan:** `eva02_small_patch14_224.mim_in22k_ft_in1k` does **not
  exist** in timm 1.0.27 → valid 224 tag is `eva02_small_patch14_224.mim_in22k` (MIM-only; head
  replaced anyway). `convnext_small.fb_in22k_ft_in1k_384` confirmed valid. Committed ensemble:
  `[convnext_small, effv2s, vit_base]` (clean 224/300-native trio); eva02_small kept as alt.

### Two real bugs found & fixed during the smoke (not just smoke artifacts)
1. **MPS autocast crash.** `torch.autocast(device_type='mps')` raises on torch 2.1.2 *even when
   `enabled=False`*. Fix: only enter an autocast context on CUDA; use `nullcontext()` otherwise.
2. **EMA cold-start = 0% val acc.** First run showed train loss falling (4.9→3.06) but val_acc
   **stuck at 0.0** — because eval uses the EMA model and `ema_decay=0.9998` implies a ~5,000-step
   averaging window, but our *entire* training is only ~1,080 steps (27 steps/ep × 40 ep), so the
   EMA never left random init. Fix: switched `ModelEmaV2`→`ModelEmaV3(use_warmup=True, foreach=False)`
   — warmup ramps effective decay (~0.80 @ step10 → ~0.99 @ step1080), and `foreach=False` avoids
   `_foreach_lerp_` which is unimplemented on MPS. **Lesson for the full run:** the handoff's
   0.9998 is an ImageNet-scale value; warmup makes it self-correct for our tiny step budget.

### Build-order step 1 — label-map gate: **PASSED**
Single fold, ConvNeXt-S, 10-class subset, 5 epochs, MPS, mixup off (overfit-style sanity):

| epoch | train loss | EMA val_acc (10 classes; random=0.10) |
|-------|-----------|----------------------------------------|
| 0 | 4.929 | 0.000 |
| 1 | 4.909 | 0.050 |
| 2 | 4.087 | 0.150 |
| 3 | 3.153 | 0.250 |
| 4 | 2.352 | **0.450** |

Val acc rises monotonically well above random while loss falls → **label wiring is correct** and the
train/eval/EMA/ckpt/OOF loop works. End-to-end `predict.py` then produced a **format-valid
`submission.csv`** (1,000 rows, header `ID,Label`, labels∈0..99, IDs match template). MixUp/
SoftTargetCE path also smoke-tested separately (no errors). Throwaway 10-class checkpoints deleted.

**Env confirmed:** torch 2.1.2 / torchvision 0.16.2 / timm 1.0.27 / sklearn 1.3.0 / numpy 1.24.4 /
pandas 2.3.2 / pillow 11.0.0, Python 3.11.5, MPS. All 1,079 train + 1,036 test images are RGB,
zero corrupt. Folds: 4×863/216 + 1×864/215 (the 4-image class triggers the expected sklearn warning).

### Next (build-order steps 2→6, to run on Kaggle CUDA)
Full 5-fold ConvNeXt-S run (real epochs) → measure OOF → add TTA → add MixUp/EMA/label-smoothing
deltas → add effv2s + vit_base ensemble → seed averaging. Log each measured OOF delta here.

---

## Kaggle handoff prep + ViT tag confirmed (2026-06-01)

### ViT ensemble member — exact tag confirmed
Locked the transformer member to **`vit_base_patch16_224.augreg_in21k_ft_in1k`**. Verified in timm
1.0.27: `input_size=(3,224,224)` (**224-native**, not 384), AugReg recipe, **ImageNet-21k pretrain →
ft_in1k** — consistent with the 21k strategy. Checked the alternatives
(`augreg2_in21k_ft_in1k`, `orig_in21k_ft_in1k`); `augreg` is the canonical, best-documented choice.
(`num_classes=1000` is just the pretrain head, replaced by our 100-way head.)

### Warmup-EMA confirmed to carry into full runs (not smoke-only)
`model.create_ema()` is hard-wired to `ModelEmaV3(use_warmup=True, foreach=False)` and `train.py`
always passes `step=global_step`; `config.train.ema=true`. So the cold-start fix applies to every
run, including the Kaggle 5-fold. `ema_decay=0.9998` is left as an upper *ceiling* the warmup schedule
never reaches at our ~1,080-step budget (eff. decay ~0.99 at the end) — intentional.

### Notebook scoped to the ConvNeXt-S baseline first
`model.ipynb` rewritten so **step 1 trains ConvNeXt-S 5-fold alone** and prints the leak-free OOF
accuracy (a dedicated cell reads `checkpoints/convnext_small/oof.npz`). The full-ensemble training +
`predict.py` cell is commented and explicitly gated "only after the baseline is approved," matching
the incremental build order.

### Repo hygiene + first commit
Verified `.gitignore` excludes `checkpoints/`, `*.pt`, `*.npz`, `logs/`, `submission.csv`, and
`Data/` (weights → Google Drive, not git; competition data is mounted on Kaggle). Initialized the git
repo and made the first commit (18 tracked files: `src/`, `config.yaml`, `model.ipynb`, `README.md`,
`requirements.txt`, docs). No data/weights staged.

### Blocker / division of labor
This dev environment has **no `gh`, no `kaggle` CLI, no Kaggle credentials, and no git remote**, so the
**GitHub push** and the **Kaggle GPU run** require the user's accounts and are theirs to execute.
Per the user's call, the baseline OOF comes from **Kaggle CUDA (the reproducible source of truth)** —
local MPS validation is intentionally skipped (a local OOF wouldn't track CUDA closely enough to
trust). Next: user creates/pushes the public repo, runs notebook step 1, and reports the ConvNeXt-S
5-fold OOF to log here as the baseline before any further layer is added.

---

## Implementation log

*(Add dated subsections below as you build.)*

### Environment

| Item | Value |
|------|-------|
| Python | 3.11.5 (local dev) |
| PyTorch | 2.1.2 (local MPS); Kaggle base-image build for reported runs |
| torchvision | 0.16.2 |
| timm | 1.0.27 (pinned) |
| Device | local: Apple M2 Pro / MPS (scratch); reported: Kaggle CUDA |
| Random seed | 1337 (`config.yaml` → `seed`) |

### Data pipeline

| Decision | Choice | Rationale | Date |
|----------|--------|-----------|------|
| Label map | explicit `int(folder)` + hard assert `"0"→0…"99"→99` | folder name IS the label; never ImageFolder alpha order | 2026-06-01 |
| Train/val split | StratifiedKFold(5, shuffle, seed=1337); OOF over all train | leak-free CV; uses all data; 4-img class warns (expected) | 2026-06-01 |
| Resize | per-model native via `timm.resolve_data_config` (224 / 300 / 384) | correct crop_pct + interpolation per checkpoint | 2026-06-01 |
| Normalization | per-checkpoint mean/std from timm (convnext .485/.456/.406; effv2s/vit .5) | match pretraining stats | 2026-06-01 |
| Augmentations | RRC(0.4–1.0)+hflip+RandAug(m9)+color-jitter+RandomErasing(0.25); MixUp/CutMix(switch0.5)+LS0.1 | main overfit defense at ~11 img/class | 2026-06-01 |

### Model & training

| Run / version | Backbone | Pretrained weights | Head changes | Freeze strategy | LR | Batch | Epochs | Val acc | Kaggle | Notes | Date |
|---------------|----------|-------------------|--------------|-----------------|----|----|--------|---------|--------|-------|------|
| step1-gate (10-class smoke) | convnext_small | fb_in22k_ft_in1k | new 100-way + dropout 0.2 | full FT + LLRD 0.8 | 1e-3 head (LLRD) | 32 | 5 | 0.45 (10cls, EMA) | — | label-map gate PASS (local MPS, no mixup) | 2026-06-01 |

### Ablations & experiments

| Experiment | Changed | Result | Conclusion | Date |
|------------|---------|--------|------------|------|
| | | | | |

### Submission history

| Date | Kaggle score | Public LB | Notes |
|------|--------------|-----------|-------|
| | | | |

---

## Decisions & open questions

**Decided**

- **Goal = maximize test accuracy** (not just clear 60%); bonus above 60% is uncapped.
- **Rank backbones by small-data transfer fit, not raw ImageNet top-1.**
- **Bring in ImageNet-21k `timm` backbones** in addition to the torchvision shortlist.
- **EfficientNet_V2_S → run at 384** (or prefer 224-native ConvNeXt_Tiny / RegNet_Y_3_2GF).
- **Strategy:** 21k backbones → multi-arch ensemble → stratified 5-fold CV → TTA → higher res →
  heavy aug/reg → discriminative LR + warmup/cosine + EMA → class-imbalance handling.

**Open**

- Final backbone set for the ensemble (how many / which archs).
- Val split: 5-fold stratified is the plan — confirm fold count given 4-image tail classes.
- Freeze-all-then-unfreeze vs. partial unfreeze from start?
- **Verify the "2025 backbone benchmark" numbers against real sources before citing.**
- Confirm no Kaggle competition-page rule restricts pretraining to ImageNet-1k.

---

## Commands reference

```bash
# Train (fill in when script/notebook is stable)
# python train.py ...

# Inference → submission.csv (must use sample_submission IDs)
# python predict.py --template Data/sample_submission.csv --out submission.csv
```

---

## Changelog (file / repo)

| Date | Change |
|------|--------|
| 2026-06-01 | Created changelog; initial data exploration documented |
| 2026-06-01 | Added `PROJECT_OVERVIEW.md` |
| 2026-06-01 | Logged goal change (clear-60% → maximize accuracy), backbone research shortlist + 384 caveat, decision to add ImageNet-21k timm backbones, full optimization strategy, and unverified-benchmark caveat |


