**Run:** (1) 2025-10-24_18-27-16_student_mbv3s_vww96

**Goal:** First run

**Change vs prev:**

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

**Change vs prev:**
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

**Change vs prev:**
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

**Run:** (4) 2025-10-25_17-46-08_student_mbv3s_vww96

**Goal:** Better performance

**Change vs prev:**
Added alpha schedule

**Config:**
alpha = start 0.9 end 0.3
temp = 4.0

teacher = 'runs/2025-10-24_21-17-59_teacher_mbv3l_vww244/model.pt'

**Result:**
val_acc= 0.8049 (epoch 20)

no person recall: 0.85
person recall: 0.73

No improvement

Observations:
Best val acc peaks at 0.8049 around epoch 20 when alpha ≈ 0.54, then steadily erodes as alpha drops below ~0.5 (teacher weight rises).
CE and KL both fall as expected; by the end KD dominates (alpha=0.3 → 70% KL), yet val does not improve.
LR schedule looks correct (ends near zero), so late-epoch drift isn’t from an LR floor.
Confusion matrix shows recall on “person(1)” is the weak spot (FN=1010), similar to your earlier runs.

Interpretation:
Heavier KD late hurts generalization here. Given the teacher is 0.88 on the student pipeline (vs 0.94 native), pushing the student to match its softened logits too strongly likely pulls it away from the hard-label optimum at 96.
Your best accuracy occurs with moderate KD (alpha ≈ 0.5–0.6). Past that, extra teacher weight doesn’t buy gains.

**Confusion highlights:**

[[3623  636]
 [1010 2790]]

---

(on PC)
**Run:** (5) 2025-10-25_21-44-06_student_mbv3s_vww96

**Goal:** Better performance

**Change vs prev:**
Constant alpha at 0.7

**Config:**
alpha = 0.7
temp = 4.0

teacher = '2025-10-24_21-17-59_teacher_mbv3l_vww244\model.pt'

**Result:**
val_acc= 0.8102 (epoch 17)
no person recall: 0.85
person recall: 0.76

**Confusion highlights:**
[[3601  658]
 [ 922 2878]]

---

(on PC)
**Run:** (6) 2025-10-25_22-24-08_student_mbv3s_vww96

**Goal:** slight boost to accuracy?

**Change vs prev:**
Same config, but with NO augs on teacher input data

**Config:**
alpha = 0.7
temp = 4.0

teacher = '2025-10-24_21-17-59_teacher_mbv3l_vww244\model.pt'

**Result:**
val_acc= 0.8080 (epoch 20)
no person recall: 0.83
person recall: 0.76

No improvement - teacher SHOULD see same augmented images as student.

**Confusion highlights:**
[[3550  709]
 [ 921 2879]]

---

(on PC)
**Run:** (7) 2025-10-26_11-36-15_student_mbv3s_vww96

**Goal:** slight boost to accuracy?

**Change vs prev:**
reverse no augs, and set alpha -> 0.5, and T -> 2.

**Config:**
alpha = 0.5
temp = 2.0

teacher = '2025-10-24_21-17-59_teacher_mbv3l_vww244\model.pt'

**Result:**
val_acc= 0.8094 (epoch 16)
no person recall: 0.85
person recall: 0.74

no improvement

**Confusion highlights:**
[[3624  635]
 [ 996 2804]]

---

(on PC)
**Run:** (8) ?

**Goal:** slight boost to accuracy?

**Change vs prev:**
(alpha now scales kl and inverses ce - opposite of before)
alpha -> 0.7
T -> 5

**Config:**
alpha = 0.7
temp = 5.0

teacher = '2025-10-24_21-17-59_teacher_mbv3l_vww244\model.pt'

**Result:**
val_acc= 0.8094 (epoch 16)
no person recall: 0.85
person recall: 0.74

no improvement

**Confusion highlights:**
[[3624  635]
 [ 996 2804]]

---

(on PC)
**Run:** (8) 2025-10-27_15-06-43_student_mbv3s_vww96_kd_a

**Goal:** First of three test runs with new margin and confidence weighting - this one is a control (KD-A)

**Change vs prev:**
less alpha - 0.6 (so less kl)
no conf. gamma
no margin

a=0.6
conf=0
margin=0
ls=0

**Config:**
kd:
alpha: 0.6
alpha_constant: true
confidence_gamma: null
label_smoothing: 0.0
margin_weight: 0.0
teacher:
arch: mobilenet_v3_large
checkpt: runs\\2025-10-26_17-53-08_teacher_mbv3l_vww96\\model.pt
pretrained: true
teacher_input_size: 96
temperature: 2.0

**Result:**
val_acc= 0.8128 (epoch 22)
no person recall: 0.84
person recall: 0.77

less alpha sees an improvement

**Confusion highlights:**
[3589, 670],
[887, 2913]

---

(on PC)
**Run:** (9) 2025-10-27_14-26-22_student_mbv3s_vww96_kd_b

**Goal:** Second of three test runs with new margin and confidence weighting - confidence weighted KD

**Change vs prev:**
conf. gamma = 2.0

a=0.6
conf=2
margin=0
ls=0

**Config:**
kd:
alpha: 0.6
alpha_constant: true
confidence_gamma: 2.0
label_smoothing: 0.0
margin_weight: 0.0
teacher:
arch: mobilenet_v3_large
checkpt: runs\2025-10-26_17-53-08_teacher_mbv3l_vww96\model.pt
pretrained: true
teacher_input_size: 96
temperature: 2.0

**Result:**
val_acc= 0.8146 (epoch 18)
no person recall: 0.84
person recall: 0.77

slight improvement with conf. weighting

**Confusion highlights:**
[3597, 662],
[893, 2907]

---

(on PC)
**Run:** (10) 2025-10-27_15-19-24_student_mbv3s_vww96_kd_c

**Goal:** Third of three test runs with new margin and confidence weighting - confidence weighted KD + margin KD

**Change vs prev:**
kept conf. gamma = 2.0
margin weight = 0.03

a=0.6
conf=2
margin=0.03
ls=0

**Config:**
kd:
alpha: 0.6
alpha_constant: true
confidence_gamma: 2.0
label_smoothing: 0.0
margin_weight: 0.03
teacher:
arch: mobilenet_v3_large
checkpt: runs\2025-10-26_17-53-08_teacher_mbv3l_vww96\model.pt
pretrained: true
teacher_input_size: 96
temperature: 2.0

**Result:**
val_acc= 0.8175 (epoch 18)
no person recall: 0.85
person recall: 0.78

slight improvement with conf. weighting + margin

**Confusion highlights:**
[3610, 649],
[853, 2947]

---

INSIGHTS:
confidence weighted KD and margin KD both improve generalisation

---

(on PC)
**Run:** (11) 2025-10-27_16-42-33_student_mbv3s_vww96_kd_c

**Goal:** Further 3 runs tweaking params - first, less kl

**Change vs prev:**
alpha 0.6 -> 0.55
label smoothing 0.0 -> 0.01

a=0.55
conf=2
margin=0.03
ls=0.01

**Config:**
kd:
alpha: 0.55
alpha_constant: true
confidence_gamma: 2.0
label_smoothing: 0.01
margin_weight: 0.03
teacher:
arch: mobilenet_v3_large
checkpt: runs\2025-10-26_17-53-08_teacher_mbv3l_vww96\model.pt
pretrained: true
teacher_input_size: 96
temperature: 2.0

**Result:**
val_acc= 0.8202 (epoch 20)
no person recall: 0.85
person recall: 0.77

less kl, more ce improved things

**Confusion highlights:**
[3628, 631],
[868, 2932]

---

(on PC)
**Run:** (12) 2025-10-27_16-43-53_student_mbv3s_vww96_kd_c

**Goal:** Further 3 runs tweaking params - next, more confidence weight

**Change vs prev:**
alpha 0.55 -> 0.6
conf 2.0 -> 2.5

a=0.6
conf=2.5
margin=0.03
ls=0.01

**Config:**
kd:
alpha: 0.6
alpha_constant: true
confidence_gamma: 2.5
label_smoothing: 0.01
margin_weight: 0.03
teacher:
arch: mobilenet_v3_large
checkpt: runs\2025-10-26_17-53-08_teacher_mbv3l_vww96\model.pt
pretrained: true
teacher_input_size: 96
temperature: 2.0

**Result:**
val_acc= 0.8196 (epoch 22)
no person recall: 0.84
person recall: 0.79

more confidence weighted KL improved performance

**Confusion highlights:**
[3592, 667],
[815, 2985]

---

(on PC)
**Run:** (13) runs\2025-10-27_17-45-28_student_mbv3s_vww96

**Goal:** Further 3 runs tweaking params - finally, less margin

**Change vs prev:**
conf 2.5 -> 2.0
margin 0.03 -> 0.02

a=0.6
conf=2
margin=0.02
ls=0.01

**Config:**
kd:
alpha: 0.6
alpha_constant: true
confidence_gamma: 2.0
label_smoothing: 0.01
margin_weight: 0.02
teacher:
arch: mobilenet_v3_large
checkpt: runs\2025-10-26_17-53-08_teacher_mbv3l_vww96\model.pt
pretrained: true
teacher_input_size: 96
temperature: 2.0

**Result:**
val_acc= 0.8154 (epoch 23)
no person recall: 0.84
person recall: 0.77

Less margin did not improve

**Confusion highlights:**
[3592, 667],
[815, 2985]

---

INSIGHTS:
Run 10 best accuracy but run 11 sees better recall balance.
lower margin underperforms
slightly less kl helped accuracy
confidence weighting helps but is sensitive to alpha - 2.5 weighting and 0.6 alpha didn't beat 2.0/0.55

What to try:
lower alpha further?
try 0.55/2.5

Best config is 0.55/2.0

---

**Run:** (14) 2025-10-27_19-22-37_student_mbv3s_vww96
alpha 0.55
conf. gamma 2.5

0.8183
worse performance

---

**Run:** (15) 2025-10-27_19-50-32_student_mbv3s_vww96
alpha 0.52
conf. gamma 2.0

0.8166
worse performance

---

(on PC)
**Run:** (16) 2025-10-27_20-32-46_student_mbv3s_vww96

**Goal:** try margin scheduler on run 11 config (0.8202)

**Change vs prev:**
0.55/2.0 with stronger inital margin with decay to 0 at epoch 20

a=0.55
conf=2
margin start= 0.04
margin end= 0
margin decay end epoc= 20 (25 epoch run)
ls=0.01

**Config:**
kd:
alpha: 0.55
alpha_constant: true
confidence_gamma: 2.0
label_smoothing: 0.01
margin_weight: 0.0
margin_weight_decay_end_epoch: 20
margin_weight_end: 0.0
margin_weight_start: 0.04
teacher:
arch: mobilenet_v3_large
checkpt: runs\2025-10-26_17-53-08_teacher_mbv3l_vww96\model.pt
pretrained: true
teacher_input_size: 96
temperature: 2.0

**Result:**
val_acc= 0.8141 (epoch 17)
no person recall: 0.84
person recall: 0.78

No improvement.

**Confusion highlights:**
[[3566  693]
 [ 851 2949]]

---

(on PC)
**Run:** (17)

**Goal:** test performance on imagenet PRETRAINED weights

**Change vs prev:**
Using pretrained weights (baseline got to 0.8486)
All other config taken from best prev run (run 11)

**Config:**
kd:
alpha: 0.55
alpha_constant: true
confidence_gamma: 2.0
label_smoothing: 0.01
margin_weight: 0.0
margin_weight_decay_end_epoch: 20
margin_weight_end: 0.0
margin_weight_start: 0.04
teacher:
arch: mobilenet_v3_large
checkpt: runs\2025-10-26_17-53-08_teacher_mbv3l_vww96\model.pt
pretrained: true
teacher_input_size: 96
temperature: 2.0

**Result:**
val_acc=
no person recall:
person recall:

No improvement.

**Confusion highlights:**
[[3566  693]
 [ 851 2949]]

---
