import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def kd_loss(student_logits, teacher_logits, targets, alpha, T, label_smoothing=0.0):
    """ Knowledge Distillation Loss """

    ce = F.cross_entropy(student_logits, targets, label_smoothing=label_smoothing)

    kl = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T**2)

    kd_loss = alpha * kl + (1 - alpha) * ce

    return kd_loss, ce, kl


def kd_train_one_epoch(
    student,
    teacher,
    loader,
    device,
    optimizer,
    scheduler,
    scaler,
    autocast,
    alpha,
    T,
    grad_clip_norm,
    label_smoothing=0.0, # shouldn't really use label smoothing with kd
):
    """ Train the student model for one epoch with knowledge distillation """
    
    student.train()
    teacher.eval()
    total, correct = 0, 0
    loss_sum, ce_sum, kl_sum = 0.0, 0.0, 0.0
    nonblock = torch.cuda.is_available()

    for imgs, labels in loader:
        
        imgs, labels = imgs.to(device, non_blocking=nonblock), labels.to(
            device, non_blocking=nonblock
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            t_in = F.interpolate(imgs, size=224, mode='bilinear', align_corners=False)
            teacher_logits = teacher(t_in)
        
        # Forward
        with autocast:
            student_logits = student(imgs)
            loss, ce_loss, kl_loss = kd_loss(
                student_logits,
                teacher_logits,
                labels,
                alpha,
                T,
                label_smoothing,
            )

        # Backward
        if scaler: # CUDA
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(student.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else: # MPS and CPU
            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                clip_grad_norm_(student.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler:
            scheduler.step()
        
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        ce_sum += ce_loss.item() * batch_size
        kl_sum += kl_loss.item() * batch_size
        correct += (student_logits.argmax(1) == labels).sum().item()
        total += batch_size
    
    tr_loss = loss_sum / total
    tr_acc = correct / total
    tr_ce = ce_sum / total
    tr_kl = kl_sum / total

    return tr_loss, tr_acc, tr_ce, tr_kl
