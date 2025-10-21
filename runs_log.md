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
lower accuracy in epoch 15 so overfit? maybe just start of oscillation?

**Confusion highlights:**

---

**Run:** (4) 2025-10-12_21-32-58_mbv3-baseline

**Goal:** Improve accuracy further

**Change vs prev:**
Changed scheduler to OneCycleLR
Tweaked random erasing (scale 0.08->0.12)
More epoches due to OneCycleLR (30)

**Config:**
96×96
bs=256
30 epochs
lr=1e-3
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=1e-3, pct_start=0.10, div_factor=25.0, final_div_factor=1e4, anneal_strategy="cos"
seed=42

**Data:**
Dataset mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc=0.8121 @ epoch 19
oscillated after epoch 19 without improving

no person recall: 0.84
person recall: 0.76

**Confusion highlights:**

---

**Run:** (5) 2025-10-13_09-39-05_mbv3-baseline

**Goal:** Attempt to reduce oscillation and improve person recall

**Change vs prev:**
OneCycleLR paramters: pct_start -> 0.05, div_factor -> 20.0

**Config:**
96×96
bs=256
30 epochs
lr=1e-3
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=1e-3, pct_start=0.05, div_factor=20.0, final_div_factor=1e4, anneal_strategy="cos"
seed=42

**Data:**
Dataset mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc=0.8118 @ epoch 18

no person recall: 0.83
person recall: 0.77

**Confusion highlights:**

---

**Run:** (6) 2025-10-19_16-06-50_mbv3_small_vww96

**Goal:** Change of code structure - confirm performance

**Change vs prev:**
Code structure
and random erasing 0.12->0.08

**Config:**
96×96
bs=256
30 epochs
lr=1e-3
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=1e-3, pct_start=0.05, div_factor=20.0, final_div_factor=1e4, anneal_strategy="cos"
seed=42

**Data:**
Dataset mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3))

**Result:**
val_acc=0.8093 @ epoch 14

no person recall: 0.86
person recall: 0.74

**Confusion highlights:**
[[3666  593]
 [ 999 2801]]

---

**Run:** (7) runs/2025-10-19_17-37-39_mbv3_small_vww96

**Goal:** Same as last - fixed confusion matrix error, and change random erasing back to 0.12

**Change vs prev:**
random erasing back 0.08->0.12

**Config:**
96×96
bs=256
30 epochs
lr=1e-3
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=1e-3, pct_start=0.05, div_factor=20.0, final_div_factor=1e4, anneal_strategy="cos"
seed=42

**Data:**
Dataset mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc=0.8114 @ epoch 18

no person recall: 0.83
person recall: 0.77

**Confusion highlights:**
[[3535  724]
 [ 878 2922]]

---

**Run:** (8) runs/2025-10-20_22-27-02_mbv3_small_vww96

**Goal:** Test performance on more stable data

**Change vs prev:**
Made change to how vww dataset is generated - vww96 now keeps aspect ratio and uses padding.
Changed to imagenet mean/std. I will use these values throughout the project for simplicity/stability.
No changes otherwise

**Config:**
96×96
bs=256
30 epochs
lr=1e-3
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=1e-3, pct_start=0.05, div_factor=20.0, final_div_factor=1e4, anneal_strategy="cos"
seed=42

**Data:**
Dataset mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc=0.8083 @ epoch 19

no person recall: 0.82
person recall: 0.77

Instability while training

**Confusion highlights:**
[[3489  770]
 [ 868 2932]]

---

**Run:** (9)

**Goal:** Address instability/oscillation, and recall

**Change vs prev:**
changed lr (and max_lr) from 0.001 to 0.0005 to reduce oscillation
added gradient norm clipping to also improve osc
added sampler to address class imbalance (recall problems)
color jitter 0.2 -> 0.15

**Config:**
96×96
bs=256
30 epochs
lr=0.0005
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=0.0005, pct_start=0.05, div_factor=20.0, final_div_factor=1e4, anneal_strategy="cos"
seed=42

**Data:**
Dataset mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.0)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc= @ epoch

no person recall:
person recall:

**Confusion highlights:**

---
