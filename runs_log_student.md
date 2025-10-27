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
**Run:** (8)

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

kd_a

(.venv) C:\Users\Thomas\Dev\repos\vww>python -m src.kd --config_path src\config\student_mbv3s_vww96_kd_a.yaml
[1/30] alpha 0.600 | train loss 0.7414 (ce 0.6485, kl 0.8034) acc 0.6209 | val loss 0.7083 acc 0.5285 | epoch time 61.3s | elapsed time 1.0m
[2/30] alpha 0.600 | train loss 0.5941 (ce 0.5616, kl 0.6159) acc 0.7180 | val loss 0.7836 acc 0.6460 | epoch time 62.2s | elapsed time 2.1m
[3/30] alpha 0.600 | train loss 0.5134 (ce 0.5112, kl 0.5149) acc 0.7523 | val loss 0.5379 acc 0.7449 | epoch time 61.9s | elapsed time 3.1m
[4/30] alpha 0.600 | train loss 0.4642 (ce 0.4773, kl 0.4555) acc 0.7744 | val loss 0.5502 acc 0.7481 | epoch time 62.0s | elapsed time 4.1m
[5/30] alpha 0.600 | train loss 0.4346 (ce 0.4572, kl 0.4196) acc 0.7852 | val loss 0.4985 acc 0.7753 | epoch time 61.8s | elapsed time 5.2m
[6/30] alpha 0.600 | train loss 0.4126 (ce 0.4418, kl 0.3931) acc 0.7945 | val loss 0.4736 acc 0.7752 | epoch time 62.0s | elapsed time 6.2m
[7/30] alpha 0.600 | train loss 0.3953 (ce 0.4283, kl 0.3733) acc 0.8014 | val loss 0.5216 acc 0.7668 | epoch time 62.6s | elapsed time 7.2m
[8/30] alpha 0.600 | train loss 0.3811 (ce 0.4179, kl 0.3565) acc 0.8076 | val loss 0.4598 acc 0.7917 | epoch time 62.4s | elapsed time 8.3m
[9/30] alpha 0.600 | train loss 0.3675 (ce 0.4071, kl 0.3410) acc 0.8136 | val loss 0.4576 acc 0.7939 | epoch time 62.4s | elapsed time 9.3m
[10/30] alpha 0.600 | train loss 0.3583 (ce 0.4006, kl 0.3302) acc 0.8182 | val loss 0.4338 acc 0.8054 | epoch time 63.0s | elapsed time 10.4m
[11/30] alpha 0.600 | train loss 0.3482 (ce 0.3926, kl 0.3186) acc 0.8227 | val loss 0.4351 acc 0.8020 | epoch time 62.5s | elapsed time 11.4m
[12/30] alpha 0.600 | train loss 0.3373 (ce 0.3836, kl 0.3064) acc 0.8277 | val loss 0.4453 acc 0.8008 | epoch time 62.5s | elapsed time 12.5m
[13/30] alpha 0.600 | train loss 0.3267 (ce 0.3743, kl 0.2949) acc 0.8320 | val loss 0.4346 acc 0.8027 | epoch time 62.6s | elapsed time 13.5m
[14/30] alpha 0.600 | train loss 0.3184 (ce 0.3671, kl 0.2859) acc 0.8369 | val loss 0.4299 acc 0.8079 | epoch time 62.0s | elapsed time 14.5m
[15/30] alpha 0.600 | train loss 0.3099 (ce 0.3591, kl 0.2770) acc 0.8411 | val loss 0.4217 acc 0.8073 | epoch time 60.6s | elapsed time 15.5m
[16/30] alpha 0.600 | train loss 0.3002 (ce 0.3510, kl 0.2663) acc 0.8450 | val loss 0.4249 acc 0.8075 | epoch time 65.2s | elapsed time 16.6m
[17/30] alpha 0.600 | train loss 0.2910 (ce 0.3422, kl 0.2569) acc 0.8506 | val loss 0.4400 acc 0.8070 | epoch time 69.0s | elapsed time 17.8m
[18/30] alpha 0.600 | train loss 0.2828 (ce 0.3359, kl 0.2473) acc 0.8532 | val loss 0.4241 acc 0.8133 | epoch time 65.6s | elapsed time 18.9m
[19/30] alpha 0.600 | train loss 0.2745 (ce 0.3266, kl 0.2398) acc 0.8588 | val loss 0.4359 acc 0.8095 | epoch time 65.3s | elapsed time 20.0m
[20/30] alpha 0.600 | train loss 0.2644 (ce 0.3175, kl 0.2290) acc 0.8636 | val loss 0.4373 acc 0.8077 | epoch time 66.0s | elapsed time 21.1m
[21/30] alpha 0.600 | train loss 0.2568 (ce 0.3108, kl 0.2209) acc 0.8680 | val loss 0.4323 acc 0.8106 | epoch time 65.8s | elapsed time 22.2m
[22/30] alpha 0.600 | train loss 0.2484 (ce 0.3018, kl 0.2128) acc 0.8738 | val loss 0.4375 acc 0.8120 | epoch time 65.5s | elapsed time 23.2m
[23/30] alpha 0.600 | train loss 0.2428 (ce 0.2966, kl 0.2070) acc 0.8757 | val loss 0.4401 acc 0.8125 | epoch time 67.2s | elapsed time 24.4m
[24/30] alpha 0.600 | train loss 0.2379 (ce 0.2911, kl 0.2025) acc 0.8793 | val loss 0.4369 acc 0.8070 | epoch time 74.6s | elapsed time 25.6m
[25/30] alpha 0.600 | train loss 0.2314 (ce 0.2847, kl 0.1958) acc 0.8834 | val loss 0.4433 acc 0.8084 | epoch time 77.8s | elapsed time 26.9m
[26/30] alpha 0.600 | train loss 0.2274 (ce 0.2814, kl 0.1914) acc 0.8855 | val loss 0.4435 acc 0.8079 | epoch time 78.4s | elapsed time 28.2m
[27/30] alpha 0.600 | train loss 0.2251 (ce 0.2785, kl 0.1896) acc 0.8874 | val loss 0.4449 acc 0.8064 | epoch time 78.1s | elapsed time 29.5m
[28/30] alpha 0.600 | train loss 0.2233 (ce 0.2765, kl 0.1879) acc 0.8878 | val loss 0.4469 acc 0.8083 | epoch time 79.1s | elapsed time 30.8m
[29/30] alpha 0.600 | train loss 0.2196 (ce 0.2726, kl 0.1842) acc 0.8892 | val loss 0.4472 acc 0.8082 | epoch time 78.5s | elapsed time 32.1m
[30/30] alpha 0.600 | train loss 0.2192 (ce 0.2714, kl 0.1844) acc 0.8913 | val loss 0.4475 acc 0.8079 | epoch time 78.8s | elapsed time 33.5m

VALIDATION SUMMARY

Best checkpoint: val acc = 0.8133 (epoch 18) (runs\2025-10-26_23-56-43_student_mbv3s_vww96_kd_a\model.pt)
Total training time: 33.5mins (2007.5s)

Confusion matrix:
[[3588  671]
 [ 877 2923]]
Key:
[['True neg (pred=0)' 'False pos (pred=1)']
 ['False neg (pred=0)' 'True pos (pred=1)']]

Classification report:
precision recall f1-score support

no_person(0) 0.80 0.84 0.82 4259
person(1) 0.81 0.77 0.79 3800

    accuracy                           0.81      8059

macro avg 0.81 0.81 0.81 8059
weighted avg 0.81 0.81 0.81 8059

---

kd_b

(.venv) C:\Users\Thomas\Dev\repos\vww>python -m src.kd --config_path src\config\student_mbv3s_vww96_kd_b.yaml
[1/30] alpha 0.600 | train loss 0.6133 (ce 0.6519, kl 0.5875) acc 0.6203 | val loss 0.7103 acc 0.5285 | epoch time 68.1s | elapsed time 1.1m
[2/30] alpha 0.600 | train loss 0.4860 (ce 0.5668, kl 0.4322) acc 0.7156 | val loss 0.5798 acc 0.7219 | epoch time 68.4s | elapsed time 2.3m
[3/30] alpha 0.600 | train loss 0.4248 (ce 0.5214, kl 0.3605) acc 0.7494 | val loss 0.6545 acc 0.7171 | epoch time 66.2s | elapsed time 3.4m
[4/30] alpha 0.600 | train loss 0.3846 (ce 0.4893, kl 0.3148) acc 0.7699 | val loss 0.4974 acc 0.7676 | epoch time 65.4s | elapsed time 4.5m
[5/30] alpha 0.600 | train loss 0.3593 (ce 0.4685, kl 0.2865) acc 0.7820 | val loss 0.5123 acc 0.7661 | epoch time 65.6s | elapsed time 5.6m
[6/30] alpha 0.600 | train loss 0.3400 (ce 0.4515, kl 0.2657) acc 0.7913 | val loss 0.4561 acc 0.7898 | epoch time 65.7s | elapsed time 6.7m
[7/30] alpha 0.600 | train loss 0.3241 (ce 0.4363, kl 0.2493) acc 0.7999 | val loss 0.4914 acc 0.7763 | epoch time 65.5s | elapsed time 7.8m
[8/30] alpha 0.600 | train loss 0.3104 (ce 0.4238, kl 0.2348) acc 0.8061 | val loss 0.4577 acc 0.7970 | epoch time 67.4s | elapsed time 8.9m
[9/30] alpha 0.600 | train loss 0.2994 (ce 0.4134, kl 0.2234) acc 0.8127 | val loss 0.4577 acc 0.7982 | epoch time 69.8s | elapsed time 10.0m
[10/30] alpha 0.600 | train loss 0.2897 (ce 0.4037, kl 0.2137) acc 0.8182 | val loss 0.4456 acc 0.8013 | epoch time 78.2s | elapsed time 11.3m
[11/30] alpha 0.600 | train loss 0.2807 (ce 0.3953, kl 0.2044) acc 0.8224 | val loss 0.4478 acc 0.7984 | epoch time 78.0s | elapsed time 12.6m
[12/30] alpha 0.600 | train loss 0.2717 (ce 0.3856, kl 0.1957) acc 0.8283 | val loss 0.4340 acc 0.8089 | epoch time 78.2s | elapsed time 13.9m
[13/30] alpha 0.600 | train loss 0.2634 (ce 0.3762, kl 0.1882) acc 0.8334 | val loss 0.4310 acc 0.8079 | epoch time 78.7s | elapsed time 15.3m
[14/30] alpha 0.600 | train loss 0.2542 (ce 0.3663, kl 0.1794) acc 0.8395 | val loss 0.4282 acc 0.8088 | epoch time 78.4s | elapsed time 16.6m
[15/30] alpha 0.600 | train loss 0.2483 (ce 0.3598, kl 0.1740) acc 0.8418 | val loss 0.4212 acc 0.8087 | epoch time 79.0s | elapsed time 17.9m
[16/30] alpha 0.600 | train loss 0.2405 (ce 0.3507, kl 0.1670) acc 0.8484 | val loss 0.4489 acc 0.7980 | epoch time 70.5s | elapsed time 19.1m
[17/30] alpha 0.600 | train loss 0.2324 (ce 0.3411, kl 0.1599) acc 0.8521 | val loss 0.4642 acc 0.8023 | epoch time 65.7s | elapsed time 20.2m
[18/30] alpha 0.600 | train loss 0.2244 (ce 0.3320, kl 0.1526) acc 0.8574 | val loss 0.4425 acc 0.8041 | epoch time 66.0s | elapsed time 21.3m
[19/30] alpha 0.600 | train loss 0.2184 (ce 0.3248, kl 0.1475) acc 0.8617 | val loss 0.4373 acc 0.8098 | epoch time 71.3s | elapsed time 22.4m
[20/30] alpha 0.600 | train loss 0.2108 (ce 0.3158, kl 0.1408) acc 0.8663 | val loss 0.4626 acc 0.8031 | epoch time 66.5s | elapsed time 23.6m
[21/30] alpha 0.600 | train loss 0.2035 (ce 0.3068, kl 0.1347) acc 0.8718 | val loss 0.4499 acc 0.8075 | epoch time 65.9s | elapsed time 24.7m
[22/30] alpha 0.600 | train loss 0.1959 (ce 0.2971, kl 0.1284) acc 0.8765 | val loss 0.4600 acc 0.8039 | epoch time 66.0s | elapsed time 25.8m
[23/30] alpha 0.600 | train loss 0.1915 (ce 0.2907, kl 0.1253) acc 0.8805 | val loss 0.4738 acc 0.8026 | epoch time 65.5s | elapsed time 26.8m
[24/30] alpha 0.600 | train loss 0.1868 (ce 0.2848, kl 0.1214) acc 0.8838 | val loss 0.4632 acc 0.8052 | epoch time 65.6s | elapsed time 27.9m
[25/30] alpha 0.600 | train loss 0.1826 (ce 0.2797, kl 0.1179) acc 0.8862 | val loss 0.4660 acc 0.8041 | epoch time 66.7s | elapsed time 29.0m
[26/30] alpha 0.600 | train loss 0.1775 (ce 0.2726, kl 0.1140) acc 0.8908 | val loss 0.4734 acc 0.8020 | epoch time 66.3s | elapsed time 30.2m
[27/30] alpha 0.600 | train loss 0.1759 (ce 0.2704, kl 0.1129) acc 0.8923 | val loss 0.4743 acc 0.8015 | epoch time 65.6s | elapsed time 31.2m
[28/30] alpha 0.600 | train loss 0.1732 (ce 0.2674, kl 0.1105) acc 0.8931 | val loss 0.4790 acc 0.7996 | epoch time 66.1s | elapsed time 32.3m
[29/30] alpha 0.600 | train loss 0.1729 (ce 0.2668, kl 0.1103) acc 0.8934 | val loss 0.4782 acc 0.8016 | epoch time 66.0s | elapsed time 33.5m
[30/30] alpha 0.600 | train loss 0.1723 (ce 0.2649, kl 0.1105) acc 0.8947 | val loss 0.4786 acc 0.8017 | epoch time 65.9s | elapsed time 34.5m

VALIDATION SUMMARY

Best checkpoint: val acc = 0.8098 (epoch 19) (runs\2025-10-27_00-11-58_student_mbv3s_vww96_kd_b\model.pt)
Total training time: 34.5mins (2072.9s)

Confusion matrix:
[[3554  705]
 [ 893 2907]]
Key:
[['True neg (pred=0)' 'False pos (pred=1)']
 ['False neg (pred=0)' 'True pos (pred=1)']]

Classification report:
precision recall f1-score support

no_person(0) 0.80 0.83 0.82 4259
person(1) 0.80 0.77 0.78 3800

    accuracy                           0.80      8059

macro avg 0.80 0.80 0.80 8059
weighted avg 0.80 0.80 0.80 8059

---

kb_c

(.venv) C:\Users\Thomas\Dev\repos\vww>python -m src.kd --config_path src\config\student_mbv3s_vww96_kd_c.yaml
[1/30] alpha 0.600 | train loss 0.9018 (ce 0.6570, kl 0.5840, margin 9.6215) acc 0.6201 | val loss 0.7036 acc 0.5285 | epoch time 83.8s | elapsed time 1.4m
[2/30] alpha 0.600 | train loss 0.6987 (ce 0.5724, kl 0.4261, margin 7.1357) acc 0.7166 | val loss 0.6275 acc 0.7131 | epoch time 77.9s | elapsed time 2.7m
[3/30] alpha 0.600 | train loss 0.6008 (ce 0.5254, kl 0.3528, margin 5.9658) acc 0.7504 | val loss 0.6139 acc 0.7307 | epoch time 79.3s | elapsed time 4.0m
[4/30] alpha 0.600 | train loss 0.5418 (ce 0.4936, kl 0.3101, margin 5.2773) acc 0.7701 | val loss 0.5247 acc 0.7588 | epoch time 79.3s | elapsed time 5.3m
[5/30] alpha 0.600 | train loss 0.5057 (ce 0.4733, kl 0.2843, margin 4.8612) acc 0.7803 | val loss 0.4957 acc 0.7796 | epoch time 78.6s | elapsed time 6.7m
[6/30] alpha 0.600 | train loss 0.4747 (ce 0.4559, kl 0.2620, margin 4.5072) acc 0.7914 | val loss 0.4731 acc 0.7858 | epoch time 78.3s | elapsed time 8.0m
[7/30] alpha 0.600 | train loss 0.4526 (ce 0.4412, kl 0.2470, margin 4.2627) acc 0.8003 | val loss 0.5217 acc 0.7719 | epoch time 74.5s | elapsed time 9.2m
[8/30] alpha 0.600 | train loss 0.4347 (ce 0.4304, kl 0.2345, margin 4.0631) acc 0.8052 | val loss 0.4798 acc 0.7894 | epoch time 68.8s | elapsed time 10.3m
[9/30] alpha 0.600 | train loss 0.4152 (ce 0.4176, kl 0.2213, margin 3.8472) acc 0.8122 | val loss 0.4530 acc 0.8012 | epoch time 66.1s | elapsed time 11.5m
[10/30] alpha 0.600 | train loss 0.4024 (ce 0.4101, kl 0.2121, margin 3.7037) acc 0.8154 | val loss 0.4319 acc 0.8052 | epoch time 67.6s | elapsed time 12.6m
[11/30] alpha 0.600 | train loss 0.3871 (ce 0.3993, kl 0.2020, margin 3.5400) acc 0.8228 | val loss 0.4442 acc 0.8015 | epoch time 66.5s | elapsed time 13.7m
[12/30] alpha 0.600 | train loss 0.3733 (ce 0.3904, kl 0.1925, margin 3.3867) acc 0.8267 | val loss 0.4761 acc 0.7994 | epoch time 66.0s | elapsed time 14.8m
[13/30] alpha 0.600 | train loss 0.3620 (ce 0.3820, kl 0.1852, margin 3.2701) acc 0.8320 | val loss 0.4363 acc 0.8068 | epoch time 66.1s | elapsed time 15.9m
[14/30] alpha 0.600 | train loss 0.3493 (ce 0.3727, kl 0.1767, margin 3.1388) acc 0.8365 | val loss 0.4296 acc 0.8120 | epoch time 65.6s | elapsed time 17.0m
[15/30] alpha 0.600 | train loss 0.3387 (ce 0.3651, kl 0.1696, margin 3.0302) acc 0.8404 | val loss 0.4300 acc 0.8125 | epoch time 65.3s | elapsed time 18.1m
[16/30] alpha 0.600 | train loss 0.3273 (ce 0.3563, kl 0.1625, margin 2.9105) acc 0.8458 | val loss 0.4331 acc 0.8066 | epoch time 66.5s | elapsed time 19.2m
[17/30] alpha 0.600 | train loss 0.3178 (ce 0.3491, kl 0.1563, margin 2.8138) acc 0.8503 | val loss 0.4293 acc 0.8167 | epoch time 66.4s | elapsed time 20.3m
[18/30] alpha 0.600 | train loss 0.3060 (ce 0.3410, kl 0.1483, margin 2.6862) acc 0.8543 | val loss 0.4291 acc 0.8164 | epoch time 65.9s | elapsed time 21.4m
[19/30] alpha 0.600 | train loss 0.2964 (ce 0.3319, kl 0.1427, margin 2.6001) acc 0.8593 | val loss 0.4327 acc 0.8136 | epoch time 65.8s | elapsed time 22.5m
[20/30] alpha 0.600 | train loss 0.2864 (ce 0.3243, kl 0.1362, margin 2.4983) acc 0.8641 | val loss 0.4472 acc 0.8108 | epoch time 65.7s | elapsed time 23.6m
[21/30] alpha 0.600 | train loss 0.2762 (ce 0.3154, kl 0.1301, margin 2.3996) acc 0.8676 | val loss 0.4348 acc 0.8160 | epoch time 66.3s | elapsed time 24.7m
[22/30] alpha 0.600 | train loss 0.2670 (ce 0.3078, kl 0.1243, margin 2.3102) acc 0.8728 | val loss 0.4402 acc 0.8157 | epoch time 58.7s | elapsed time 25.7m
[23/30] alpha 0.600 | train loss 0.2611 (ce 0.3031, kl 0.1205, margin 2.2529) acc 0.8762 | val loss 0.4410 acc 0.8146 | epoch time 62.6s | elapsed time 26.7m
[24/30] alpha 0.600 | train loss 0.2551 (ce 0.2971, kl 0.1171, margin 2.1989) acc 0.8790 | val loss 0.4448 acc 0.8128 | epoch time 62.7s | elapsed time 27.8m
[25/30] alpha 0.600 | train loss 0.2480 (ce 0.2911, kl 0.1128, margin 2.1306) acc 0.8821 | val loss 0.4513 acc 0.8102 | epoch time 62.6s | elapsed time 28.8m
[26/30] alpha 0.600 | train loss 0.2427 (ce 0.2862, kl 0.1096, margin 2.0805) acc 0.8853 | val loss 0.4510 acc 0.8109 | epoch time 62.6s | elapsed time 29.8m
[27/30] alpha 0.600 | train loss 0.2415 (ce 0.2855, kl 0.1089, margin 2.0672) acc 0.8849 | val loss 0.4521 acc 0.8113 | epoch time 62.6s | elapsed time 30.9m
[28/30] alpha 0.600 | train loss 0.2385 (ce 0.2831, kl 0.1068, margin 2.0396) acc 0.8863 | val loss 0.4552 acc 0.8131 | epoch time 62.4s | elapsed time 31.9m
[29/30] alpha 0.600 | train loss 0.2351 (ce 0.2795, kl 0.1050, margin 2.0086) acc 0.8891 | val loss 0.4548 acc 0.8116 | epoch time 62.6s | elapsed time 33.0m
[30/30] alpha 0.600 | train loss 0.2366 (ce 0.2807, kl 0.1061, margin 2.0214) acc 0.8882 | val loss 0.4551 acc 0.8118 | epoch time 62.4s | elapsed time 34.0m

VALIDATION SUMMARY

Best checkpoint: val acc = 0.8167 (epoch 17) (runs\2025-10-27_00-21-33_student_mbv3s_vww96_kd_c\model.pt)
Total training time: 34.0mins (2040.4s)

Confusion matrix:
[[3628  631]
 [ 886 2914]]
Key:
[['True neg (pred=0)' 'False pos (pred=1)']
 ['False neg (pred=0)' 'True pos (pred=1)']]

Classification report:
precision recall f1-score support

no_person(0) 0.80 0.85 0.83 4259
person(1) 0.82 0.77 0.79 3800

    accuracy                           0.81      8059

macro avg 0.81 0.81 0.81 8059
weighted avg 0.81 0.81 0.81 8059
