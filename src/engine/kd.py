import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_


def _logit_margin(logits, targets):
    true = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets.unsqueeze(1), False)
    other = logits.masked_fill(~mask, float('-inf')).max(dim=1).values
    return true - other


def kd_loss(
    student_logits,
    teacher_logits,
    targets,
    alpha,
    T,
    label_smoothing=0.0,
    confidence_gamma=None,
    margin_weight=0.0,
):
    """ Knowledge Distillation Loss """

    ce = F.cross_entropy(student_logits, targets, label_smoothing=label_smoothing)

    student_logp = F.log_softmax(student_logits / T, dim=1)
    teacher_prob = F.softmax(teacher_logits / T, dim=1)
    kl_per_sample = F.kl_div(student_logp, teacher_prob, reduction="none").sum(dim=1)
    kl_per_sample = kl_per_sample * (T**2)

    if confidence_gamma and confidence_gamma > 0:
        weights = teacher_prob.max(dim=1).values.pow(confidence_gamma)
        kl = (weights * kl_per_sample).mean()
    else:
        kl = kl_per_sample.mean()

    total = alpha * kl + (1 - alpha) * ce

    margin_loss = 0.0
    if margin_weight and margin_weight > 0:
        student_margin = _logit_margin(student_logits, targets)
        teacher_margin = _logit_margin(teacher_logits, targets)
        margin_loss = F.mse_loss(student_margin, teacher_margin)
        total = total + margin_weight * margin_loss

    return total, ce, kl, margin_loss


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
    teacher_input_size=None,
    confidence_gamma=None,
    margin_weight=0.0,
):
    """ Train the student model for one epoch with knowledge distillation """
    
    student.train()
    teacher.eval()
    total, correct = 0, 0
    loss_sum, ce_sum, kl_sum, margin_sum = 0.0, 0.0, 0.0, 0.0
    t_size = int(teacher_input_size) if teacher_input_size else None
    nonblock = torch.cuda.is_available()

    for imgs, labels in loader:
        
        imgs, labels = imgs.to(device, non_blocking=nonblock), labels.to(
            device, non_blocking=nonblock
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            t_in = imgs
            if t_size and t_size != imgs.shape[-1]:
                t_in = F.interpolate(imgs, size=t_size, mode="bilinear", align_corners=False)
            teacher_logits = teacher(t_in)
        
        # Forward
        with autocast:
            student_logits = student(imgs)
            loss, ce_loss, kl_loss, margin_loss = kd_loss(
                student_logits,
                teacher_logits,
                labels,
                alpha,
                T,
                label_smoothing,
                confidence_gamma,
                margin_weight,
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
        if torch.is_tensor(margin_loss):
            margin_sum += margin_loss.item() * batch_size
        else:
            margin_sum += margin_loss * batch_size
        correct += (student_logits.argmax(1) == labels).sum().item()
        total += batch_size
    
    tr_loss = loss_sum / total
    tr_acc = correct / total
    tr_ce = ce_sum / total
    tr_kl = kl_sum / total
    tr_margin = margin_sum / total if margin_sum > 0 else 0.0

    return tr_loss, tr_acc, tr_ce, tr_kl, tr_margin
