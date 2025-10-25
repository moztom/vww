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

**Run:** (2) 2025-10-24_21-17-59_teacher_mbv3l_vww244

**Goal:** re-run with bugs fixed

## **Change vs prev:**

Fixed scheduler bug (final_div_factor 20 -> 10000)
Also fixed grad clipping bug (scaling gradients to zero oops)

**Config:**
default pretrain (IMAGENET1K_V2)

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc= 0.9455 (epoch 12)

no person recall: 0.97
person recall: 0.92

**Confusion highlights:**
[[4116  143]
 [ 302 3498]]

---
