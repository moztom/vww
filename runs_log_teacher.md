**Run:** (1) 2025-10-21_15-39-10_mbv3_large_vww224

**Goal:** First run

## **Change vs prev:**

**Config:**
default pretrain (IMAGENET1K_V2)

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc= 0.939 (epoch 6)

no person recall: 0.96
person recall: 0.91

**Confusion highlights:**
[4078, 181],
[333, 3467]

---
