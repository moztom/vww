**Run:** (1) 2025-10-27_22-30-33_teacher_effnetb2_vww96

**Goal:** First run

**Change vs prev:**

**Config:**
default pretrain (IMAGENET1K_V2)
trained on 96x96

**Data:**

**Result:**
val_acc= 0.8992

no person recall: 0.91
person recall: 0.88

Room for improvement

**Confusion highlights:**

---

Run 2 - 2025-10-28_22-11-40_teacher_effnetb2_vww96

(accidently) vww96
accuracy - 0.9115

key things:
128 batch
0.0005 lr

---

Run 3 - 2025-10-28_22-12-51_teacher_effnetb2_vww96

vww96
accuracy - 0.9027

key things:
192 batch
0.00015 lr

---

Run 4 - 2025-10-29_20-34-25_teacher_effnetb2_vww224

vww224
accuracy - 0.9514
128 batch
0.0005 lr
