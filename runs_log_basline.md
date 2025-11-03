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
Imagenet mean/std normalization
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

**Run:** (9) runs/2025-10-21_11-28-36_mbv3_small_vww96

**Goal:** revert some changes - just test sampler and grad clip. Goal is better recall with similar acc to run 8.

**Change vs prev:**
lr/max_lr back to 0.001
colour jitter back to 0.2
set sampler class weight \* 0.5

**Config:**
96×96
bs=256
30 epochs
lr=0.001
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=0.001, pct_start=0.05, div_factor=20.0, final_div_factor=1e4, anneal_strategy="cos"
seed=42

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc= 0.7941 (epoch 21)

no person recall: 0.82
person recall: 0.75

recall went down. This is because i set sampler to 0.5 actually skewing the data.

**Confusion highlights:**
[[3498  761]
 [ 953 2847]]

---

**Run:** (10) runs/2025-10-21_12-49-31_mbv3_small_vww96

**Goal:** Improve recall with similar acc to run 8

**Change vs prev:**
sampler class weight -> \* 1
rm clipping for now, though this shouldn't affect it.
Only change over 8 is sampler

**Config:**
96×96
bs=256
30 epochs
lr=0.001
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=0.001, pct_start=0.05, div_factor=20.0, final_div_factor=1e4, anneal_strategy="cos"
seed=42

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc= 0.7977 (epoch 19)

no person recall: 0.83
person recall: 0.76

**Confusion highlights:**
[[3525  734]
 [ 921 2879]]

---

**Run:** (11) runs/2025-10-21_13-52-14_mbv3_small_vww96

**Goal:** Revert performance back to 8

**Change vs prev:**
rm sampler - should be back to same config as 8

**Config:**
96×96
bs=256
30 epochs
lr=0.001
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=0.001, pct_start=0.05, div_factor=20.0, final_div_factor=1e4, anneal_strategy="cos"
seed=42

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc= 0.8083 (epoch 19)

no person recall: 0.84
person recall: 0.76

**Confusion highlights:**
[[3558  701]
 [ 907 2893]]

---

**Run:** (12) runs/2025-10-23_21-24-02_baseline_mbv3s_vww96

**Goal:** Test performance with grad clipping added - reduce oscillation / improve stability

**Change vs prev:**
Added grad clipping at 1.0

**Config:**
96×96
bs=256
30 epochs
lr=0.001
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=0.001, pct_start=0.05, div_factor=20.0, final_div_factor=1e4, anneal_strategy="cos"
seed=42

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc= 0.8053 (epoch 21)

no person recall: 0.83
person recall: 0.75

More stable, and got to about the same accuracy.

**Confusion highlights:**
[[3556  703]
 [ 933 2867]]

---

**Run:** (13) runs/2025-10-23_22-35-05_baseline_mbv3s_vww96

**Goal:** reduce clipping to improve accuracy

**Change vs prev:**
grad clipping 1.0 -> 2.0 (higher is "less")

**Config:**
96×96
bs=256
30 epochs
lr=0.001
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=0.001, pct_start=0.05, div_factor=20.0, final_div_factor=1e4, anneal_strategy="cos"
seed=42

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc= 0.8079 (epoch 18)

no person recall: 0.84
person recall: 0.77

A bit less stable, better performance.

**Confusion highlights:**
[[3582  677]
 [ 878 2922]]

---

**Run:** (14) runs/2025-10-24_19-51-08_baseline_mbv3s_vww96

**Goal:** fix incorrect scheduler param. More stability, better generalisation?

**Change vs prev:**
Fixed error: final_div_factor was using div_factor value (20.0). Changed to 10000.0

**Config:**
96×96
bs=256
30 epochs
lr=0.001
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=0.001, pct_start=0.05, div_factor=20.0, final_div_factor=1e4, anneal_strategy="cos"
seed=42

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc= 0.8095 (epoch 14)

no person recall: 0.86
person recall: 0.75

Much more stable, better performance (apart from person recall)

**Confusion highlights:**
[[3652  607]
 [ 937 2863]]

---

(on PC)
**Run:** (15) 2025-10-27_21-10-40_baseline_mbv3s_vww96

**Goal:** Test performance with imagenet-pretrained weight

**Change vs prev:**
initializing the model with imagenet weights
Everything else the same

**Config:**
96×96
bs=256
30 epochs
lr=0.001
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=0.001, pct_start=0.05, div_factor=20.0, final_div_factor=1000, anneal_strategy="cos"
seed=42

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip (p=0.5)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc= 0.8486 (epoch 3)

no person recall: 0.87
person recall: 0.82

**Confusion highlights:**

---

(on PC)
**Run:** (16) 2025-10-27_21-51-09_baseline_mbv3s_vww96

**Goal:** Tweak training config to match pretrained weights

**Change vs prev:**
lr -> 0.0007
re p=0 (turned off random erasing to match student)

**Config:**
96×96
bs=256
30 epochs
lr=0.0007
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=0.0007, pct_start=0.05, div_factor=20.0, final_div_factor=1000, anneal_strategy="cos"
seed=42

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip (p=0.5)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
RandomErasing(p=0.0, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc= 0.8562 (epoch 19)

no person recall: 0.87
person recall: 0.83

**Confusion highlights:**
[[3715  544]
 [ 633 3167]]

---

(on PC)
**Run:** (17) 2025-11-03_22-00-04_baseline_mbv3s_vww96

**Goal:** Re-run with CUDA determinism, and with MACs/params count

**Change vs prev:**
no change

**Config:**
96×96
bs=256
30 epochs
lr=0.0007
Optimiser: AdamW - weight_decay=1e-4
Loss fcn: CrossEntropyLoss - label_smoothing=0.05
scheduler: OneCycleLR - max_lr=0.0007, pct_start=0.05, div_factor=20.0, final_div_factor=1000, anneal_strategy="cos"
seed=42

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip (p=0.5)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
RandomErasing(p=0.0, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc= 0.8523 (epoch 18)

no person recall: 0.87
person recall: 0.83

**Confusion highlights:**
[[3702  557]
 [ 649 3151]]

---
