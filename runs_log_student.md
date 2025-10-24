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

**Run:** (2)

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
val_acc= 0.7657 (epoch 8)

no person recall: 0.95
person recall: 0.56

failed to converge

**Confusion highlights:**
[[4035  224]
 [1671 2129]]

---
