**Run:** (1) 2025-10-24_18-27-16_student_mbv3s_vww96

**Goal:** First run

## **Change vs prev:**

**Config:**
alpha = 0.5
temp = 4.0

teacher = 'runs/2025-10-21_15-39-10_mbv3_large_vww224/model.pt'

**Result:**
val_acc= 0.7657 (epoch 8)

no person recall: 0.95
person recall: 0.56

failed to converge - likely due to teacher being given same 96x96 inputs as student

**Confusion highlights:**
[[4035  224]
 [1671 2129]]

---

**Run:** (2) 2025-10-24_21-48-50_student_mbv3s_vww96

**Goal:** Better performance

## **Change vs prev:**

Fixed scheduler bug (final_div_factor 20 -> 10000)
Upscaling teacher inputs to 224x224 resolution
Remove random erasing (p=0)

**Config:**
alpha = 0.5
temp = 4.0

teacher = 'runs/2025-10-21_15-39-10_mbv3_large_vww224/model.pt'

**Result:**
val_acc= 0.8102 (epoch 17)

no person recall: 0.87
person recall: 0.73

More epochs with more patience?

**Confusion highlights:**
[[3687  572]
 [1026 2774]]

---

**Run:** (3) 2025-10-25_10-35-10_student_mbv3s_vww96

**Goal:** Better performance

## **Change vs prev:**

New teacher
More epochs/patience (50/8)

**Config:**
alpha = 0.5
temp = 4.0

teacher = 'runs/2025-10-24_21-17-59_teacher_mbv3l_vww244/model.pt'

**Result:**
val_acc= 0.8068 (epoch 19)

no person recall: 0.84
person recall: 0.76

**Confusion highlights:**
[[3580  679]
 [ 921 2879]]

---

**Run:** (4)

**Goal:** Better performance

## **Change vs prev:**

Added alpha schedule

**Config:**
alpha = start 0.9 end 0.3
temp = 4.0

teacher = 'runs/2025-10-24_21-17-59_teacher_mbv3l_vww244/model.pt'

**Result:**
val_acc=

no person recall:
person recall:

**Confusion highlights:**

---
