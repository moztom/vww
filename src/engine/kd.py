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

    kd_loss = alpha * ce + (1 - alpha) * kl

    return kd_loss


def kd_train_one_epoch(
    model,
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
    
    model.train()
    teacher.eval()
    total, correct, loss_sum = 0, 0, 0.0
    nonblock = torch.cuda.is_available()

    for imgs, labels in loader:
        
        imgs, labels = imgs.to(device, non_blocking=nonblock), labels.to(
            device, non_blocking=nonblock
        )

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_logits = teacher(imgs)
        
        # Forward
        with autocast:
            student_logits = model(imgs)
            loss = kd_loss(student_logits, teacher_logits, labels, alpha, T, label_smoothing)

        # Backward
        if scaler: # CUDA
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else: # MPS and CPU
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler:
            scheduler.step()
        
        loss_sum += loss.item() * labels.size(0)
        correct += (student_logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    
    tr_loss = loss_sum / total
    tr_acc = correct / total
    
    return tr_loss, tr_acc
