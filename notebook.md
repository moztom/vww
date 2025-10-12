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
RandomHorizontalFlip,
dataset mean/std normalization

**Result:**
**Confusion highlights:**
**Next step:** try stronger aug, label smoothing, longer train (30e)

---
