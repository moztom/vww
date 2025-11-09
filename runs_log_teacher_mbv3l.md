**Run:** (1) 2025-10-21_15-39-10_mbv3_large_vww224

**Goal:** First run

**Change vs prev:**

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

**Change vs prev:**
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

(on PC)
**Run:** (3) 2025-10-26_17-53-08_teacher_mbv3l_vww96

**Goal:** Fine-tune on 96x96 dataset. Expect lower accuracy but ideally higher than previous on upsampled data (~0.8845 val_acc)

**Change vs prev:**
trained on vww96 rather than vww224
New config file:
src\config\teacher_mbv3l_vww96.yaml

**Config:**
default pretrain (IMAGENET1K_V2)

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.0)
RandomErasing OFF (p=0.0)

**Result:**
val_acc= 0.8862 (epoch 3)

no person recall: 0.91
person recall: 0.85

**Confusion highlights:**
[[3883  376]
 [ 573 3227]]

---

(on PC)
**Run:** (4) 2025-10-26_22-24-43_teacher_mbv3l_vww96_ft_a

**Goal:** Higher accuracy

**Change vs prev:**
lr down to 0.0002
freezed backbone for 2 epochs

config file:
src\config\teacher_mbv3l_vww96_ft_a.yaml

**Config:**
default pretrain (IMAGENET1K_V2)

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.0)
RandomErasing OFF (p=0.0)

**Result:**
val_acc= 0.8857 (epoch 4)

no person recall: 0.90
person recall: 0.86

No improvement.

**Confusion highlights:**
[[3847  412]
 [ 549 3251]]

---

(on PC)
**Run:** (5) 2025-10-26_22-34-15_teacher_mbv3l_vww96_ft_b

**Goal:** Higher accuracy

**Change vs prev:**
BN stuff, see config file

config file:
src\config\teacher_mbv3l_vww96_ft_b.yaml

**Config:**
default pretrain (IMAGENET1K_V2)

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.0)
RandomErasing OFF (p=0.0)

**Result:**
val_acc= 0.8850 (epoch 4)

no person recall: 0.90
person recall: 0.86

No improvement.

**Confusion highlights:**
[[3831  428]
 [ 524 3276]]

---

(on PC)
**Run:** (6) 2025-10-26_22-43-37_teacher_mbv3l_vww96_ft_c

**Goal:** Higher accuracy

**Change vs prev:**
Same as 5, with EMA

config file:
src\config\teacher_mbv3l_vww96_ft_c.yaml

**Config:**
default pretrain (IMAGENET1K_V2)

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.0)
RandomErasing OFF (p=0.0)

**Result:**
val_acc= 0.8852 (epoch 7)

no person recall: 0.90
person recall: 0.86

No improvement.

**Confusion highlights:**
[[3841  418]
 [ 533 3267]]

---

(on PC)
**Run:** (7) 2025-10-28_13-29-51_teacher_mbv3l_vww96_native

Training directly on 96 rather than fine-tuning from 224 to 96

**Result:**
val_acc= 0.8876 (epoch 4)

no person recall: 0.90
person recall: 0.86

---

**Run:** (8) 2025-11-03_22-37-10_teacher_mbv3l_vww244

(also see 2025-11-04_18-55-01_teacher_mbv3l_vww244)

**Goal:** re-run 224 teacher with CUDA determinism, and with MACs/params count

**Change vs prev:**
this is 224 native, identical run to run 2 but with more patience (3 -> 5)

**Config:**
default pretrain (IMAGENET1K_V2)

**Data:**
Imagenet mean/std normalization
RandomHorizontalFlip
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))

**Result:**
val_acc= 0.9425 (epoch 10)

no person recall: 0.97
person recall: 0.91

**Confusion highlights:**
[[4137  122]
 [ 341 3459]]

---

**Run:** (9) 2025-11-08_22-28-58_teacher_mbv3l_vww96

**Goal:** Same as run 3 but with the below changes

**Change vs prev:**
compared to run 3:
color_jitter (0.15 -> 0.2): [0.2, 0.2, 0.2, 0.0]
random_erasing (p=0 -> 0.15): [0.15, 0.02, 0.12, 0.3, 3.3]
kd label smoothing 0.02 -> 0.05
grad clip 0 -> 2
freeze backbone for 2 epochs
more epochs
pct_start 0.1 -> 0.3

**Config:**

**Data:**

**Result:**
val_acc=0.88695

no person recall: 0.94
person recall: 0.83

ever so slightly better results than run 3 (my current best)
next try self-distill

**Confusion highlights:**
[3986, 273],
[638, 3162]

---

**Run:** (10) 2025-11-09_00-31-36_teacher_mbv3l_vww96_self_distill

**Goal:** Try out self distillation from 224 teacher (run 2) to 96
Compare to run 3 which fine-tuned from 224 to 96. This approach uses kd instead.

**Change vs prev:**

**Config:**

**Data:**

**Result:**
val_acc= 0.8896

no person recall: 0.93
person recall: 0.84

**Confusion highlights:**
[3969, 290],
[600, 3200]

---

**Run:** (11) 2025-11-09_13-44-43_teacher_mbv3l_vww96_self_distill

**Goal:** Self distill but with slightly more random erasing

**Change vs prev:**
RE p 0.15 -> 0.2

**Config:**

**Data:**

**Result:**
val_acc= 0.8889

no person recall: 0.93
person recall: 0.85

Slightly worse accuracy but better calibration. Not good enough improvement so reverting change.
Though this one is probably better as a teacher for kd

**Confusion highlights:**
[[3948  311]
 [ 584 3216]]

---

**Run:** (12) 2025-11-09_14-29-18_teacher_mbv3l_vww96_self_distill

**Goal:** Self distill but with frozen backbone for 2 epochs

**Change vs prev:**
added frozen backbone wiring for kd
re back to 0.15

**Config:**

**Data:**

**Result:**
val_acc= 0.8888 (epoch 12)

no person recall: 0.93
person recall: 0.85

No improvement, but did stabalise it seems.

**Confusion highlights:**

---

**Run:** (13) 2025-11-09_15-42-08_teacher_mbv3l_vww96_self_distill

**Goal:** Self distill with EMA (compare to prev run)

**Change vs prev:**
add ema decay at 0.999

(with froz backbone)

**Config:**

**Data:**

**Result:**
val_acc= 0.8909 (epoch 9)

no person recall: 0.93
person recall: 0.85

Nice performance gain

**Confusion highlights:**
[[3961  298]
 [ 581 3219]]

---
