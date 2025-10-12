**Run:** (1) 2025-10-12_17-06-49_mbv3-baseline

**Goal:** Establish baseline MobileNetV3-Small on VWW96
**Change vs prev:** First run

**Config:**
96×96
AdamW
lr=3e-4
bs=256
10 epochs
CrossEntropyLoss
CosineAnnealingLR
seed=42

**Data:**
Dataset mean/std normalization
RandomHorizontalFlip

**Result:**
val_acc=0.7398 @ epoch 10
val_acc=0.7410 @ epoch 7
Overfit after epoch 7

**Confusion highlights:**

---

**Run:** (2) 2025-10-12_17-52-59_mbv3-baseline

**Goal:** Improve accuracy and reduce overfit
**Change vs prev:** 15 epochs and more data aug - (ColorJitter, RandomErasing)

**Config:**
96×96
AdamW
lr=3e-4
bs=256
15 epochs
CrossEntropyLoss
CosineAnnealingLR
seed=42

**Data:**
Dataset mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3))

**Result:**
val_acc=0.7646 @ epoch 14
lower epoch 15 so overfit

**Confusion highlights:**

---

**Run:** (3) 2025-10-12_19-13-56_mbv3-baseline

**Goal:** Improve accuracy
**Change vs prev:**
Label smoothing (0.05) on loss function
Weight decay (1e-4) on optimiser
LR 3e-4 -> 1e-3

**Config:**
96×96
bs=256
15 epochs
lr=1e-3
Optimiser: AdamW
Loss fcn: CrossEntropyLoss
scheduler: CosineAnnealingLR
seed=42

**Data:**
Dataset mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3))

**Result:**
val_acc=0.8072 @ epoch 14
lower epoch 15 so overfit

**Confusion highlights:**

---

**Run:** (4)

**Goal:** Improve accuracy further
**Change vs prev:**
Changed scheduler to OneCycleLR
Tweaked random erasing data aug (0.08->0.12)

**Config:**
96×96
bs=256
15 epochs
lr=1e-3
Optimiser: AdamW
Loss fcn: CrossEntropyLoss
scheduler: OneCycleLR
seed=42

**Data:**
Dataset mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**

**Confusion highlights:**

---
