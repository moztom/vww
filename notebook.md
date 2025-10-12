**Confusion highlights:** FNs mostly small/occluded persons

**Run:**

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

**Run:**

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
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
**Confusion highlights:**
**Next step:** try stronger aug, label smoothing, longer train (30e)

---
