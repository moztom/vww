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
**Run:** (2)

**Goal:** more fine-tune time since there seems to be more headroom

**Change vs prev:**
More fine-tuning epochs (4->10)
Also fixed a bug where the alpha and temp were not retrieved correctly so prev run was with a=0.5 and t=4. This run should use the correct a=0.55 and t=3

**Config:**
see config.yaml

**Result:**

---
