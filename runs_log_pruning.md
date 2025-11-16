(On PC)
**Run:** (1) runs\2025-11-07_21-12-20_student_mbv3s_vww96_prune

**Goal:** first run of pruning pipeline

**Change vs prev:**

**Config:**
see config.yaml

**Result:**
Total prunable channels (expand convs): 2280
Baseline accuracy: 0.8688 (eval 12.9s)
Baseline complexity: params=1,519,906 (1.52M) | MACs=12,147,832 (12.15M)

---

Pruning towards 10.00% global channel sparsity (expand convs only)
Removing 228 channels (cumulative fraction 0.0%)
Cumulative removed fraction: 10.0%
Recalibrating BatchNorm statistics using the val loader...
Post-surgery pre-recovery accuracy: 0.8345
[target 10.0% | epoch 1/4] train loss 0.1810 acc 0.9191 | val loss 0.3497 acc 0.8559 | time 55.7s
[target 10.0% | epoch 2/4] train loss 0.1505 acc 0.9359 | val loss 0.3496 acc 0.8584 | time 49.8s
[target 10.0% | epoch 3/4] train loss 0.1402 acc 0.9424 | val loss 0.3529 acc 0.8578 | time 48.8s
[target 10.0% | epoch 4/4] train loss 0.1318 acc 0.9479 | val loss 0.3515 acc 0.8598 | time 49.1s
Best recovery epoch 4: val acc 0.8598 | final eval 0.8598

---

Pruning towards 20.00% global channel sparsity (expand convs only)
Removing 228 channels (cumulative fraction 10.0%)
Cumulative removed fraction: 20.0%
Recalibrating BatchNorm statistics using the val loader...
Post-surgery pre-recovery accuracy: 0.8149
[target 20.0% | epoch 1/4] train loss 0.2396 acc 0.8903 | val loss 0.3596 acc 0.8459 | time 48.6s
[target 20.0% | epoch 2/4] train loss 0.1975 acc 0.9099 | val loss 0.3561 acc 0.8525 | time 50.9s
[target 20.0% | epoch 3/4] train loss 0.1795 acc 0.9197 | val loss 0.3545 acc 0.8517 | time 54.1s
[target 20.0% | epoch 4/4] train loss 0.1677 acc 0.9269 | val loss 0.3543 acc 0.8540 | time 48.1s
Best recovery epoch 4: val acc 0.8540 | final eval 0.8540
Stopping: accuracy 0.8540 below baseline threshold 0.8562.

**Analysis:**
Needs more fine-tuning time, maybe using val loader for bn recal is hurting too.
Should also try finer pruning steps (less jump between targets)
Could try lower lr_high_threshold (maybe 0.2) if recovery starts to jitter. This keeps 1e-4 for 10–15%, and uses 5e-5 at >=20%

---

(On PC)
**Run:** (2) 2025-11-08_16-02-58_student_mbv3s_vww96_prune

**Goal:** more fine-tune time since there seems to be more headroom

**Change vs prev:**
More fine-tuning epochs (4->10)
Also fixed a bug where the alpha and temp were not retrieved correctly so prev run was with a=0.5 and t=4. This run should use the correct a=0.55 and t=3
Also changed to newly trained baseline accuracy 0.8517 (from 0.8562)

**Config:**
see config.yaml

**Result:**
10.00% 0.8608
15.00% 0.8561
20.00% 0.8523
25.00% 0.8479

**Analysis:**
distillation strength (a=0.55) seems to have hurt performance at 20% compare to run 1
20%-25% too big of a jump.

---

(On PC)
**Run:** (3) 2025-11-08_17-51-57_student_mbv3s_vww96_prune

**Goal:** tune params to recovery better and try lower target steps (to reduce single step shock)

**Change vs prev:**
alpha = 0.5, t = 4 (from 0.55,3)
lr: 0.00007 (from 0.0001)
lr_high: 0.00003 (from 0.00005)
lr_high_threshold: 0.21 (from 0.25)
targets: [0.10, 0.12, 0.15, 0.20, 0.22, 0.25. 0.27, 0.30]
epochs 12, p=5 (from 10, 4)

Also set baseline min accuracy to 0.8400 to let pruning run without halts.

**Config:**
see config.yaml

**Result:**
10.00% 0.8605
12.00% 0.8610
15.00% 0.8547
20.00% 0.8486
22.00% 0.8490
25.00% 0.8486
27.00% 0.8474
30.00% 0.8456

**Analysis:**

---

(On PC)
**Run:** (4) 2025-11-08_19-15-45_student_mbv3s_vww96_prune

**Goal:** tune params to recovery better and try lower target steps (to reduce single step shock)

**Change vs prev:**
lr: 0.0001 from 0.00007
still switching to 0.00003 after 20%
targets: [0.10, 0.12, 0.15, 0.17, 0.20, 0.22, 0.25. 0.27, 0.30]

**Config:**
see config.yaml

**Result:**
10.00% 0.8608
12.00% 0.8598
15.00% 0.8553
17.00% 0.8562
20.00% 0.8526
22.00% 0.8499
25.00% 0.8510
27.00% 0.8480
30.00% 0.8460

**Analysis:**

---

(On PC)
**Run:** (5) 2025-11-09_18-08-14_student_mbv3s_vww96_prune

**Goal:** change to l1_norm importance

**Change vs prev:**
bn_gamma -> l1_norm

**Config:**
see config.yaml

**Result:**
10.00% 0.8657
15.00% 0.8650
20.00% 0.8620
25.00% 0.8618
27.00% 0.8633
30.00% 0.8620

**Analysis:**
Much better performance with l1_norm importance - does a better job identifying unimportant channels

---

(On PC)
**Run:** (6) 2025-11-14_20-26-41_student_mbv3s_vww96_prune

**Goal:** same, but with higher target

**Change vs prev:**
targets increment in 5% steps from 0.10 to 0.80
also decreased training epochs: 12 -> 10

**Config:**

**Result:**
10.00% 0.8654
15.00% 0.8631
20.00% 0.8613
25.00% 0.8620
30.00% 0.8621
35.00% 0.8611
40.00% 0.8599
45.00% 0.8583
50.00% 0.8602
55.00% 0.8603
60.00% 0.8597
65.00% 0.8604
70.00% 0.8531
75.00% 0.8491
80.00% n/a

**Analysis:**

---
