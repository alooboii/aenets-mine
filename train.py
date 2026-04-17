"""
train.py — Unified Knowledge Distillation Training Script
==========================================================
Supports six distillation schemes selected via --method:

  logit_kd          Classical KD (Hinton et al.)         logits_kd.py
  dkd               Decoupled KD (Zhao et al., 2022)     dkd.py
  fitnet            FitNets (Romero et al., 2015)         fitnet.py
  crd               Contrastive Repr. Distil.             crd.py
  sae_injection     SAE feature injection                 sae_injection.py
  sae_weightcompress SAE + weight compression             sae_weightcompress.py

Usage examples
--------------
# Logit KD
python train.py --method logit_kd \
    --teacher-model resnet50 --student-model resnet18 \
    --dataset CIFAR100 --epochs 30

# DKD
python train.py --method dkd \
    --teacher-model resnet50 --student-model resnet18 \
    --dataset CIFAR100 --epochs 30 \
    --dkd-alpha 1.0 --dkd-beta 8.0 --temperature 4.0

# FitNet (student adapter)
python train.py --method fitnet \
    --teacher-model resnet50 --student-model resnet18 \
    --dataset CIFAR100 --epochs 30 \
    --teacher-layers layer3:1024 --student-layers layer2:512 \
    --fitnet-adapter student

# CRD
python train.py --method crd \
    --teacher-model resnet50 --student-model resnet18 \
    --dataset CIFAR100 --epochs 30 \
    --crd-teacher-layer layer3 --crd-teacher-channels 1024 \
    --crd-student-layer layer2 --crd-student-channels 512

# SAE Injection
python train.py --method sae_injection \
    --teacher-model resnet50 --student-model resnet18 \
    --dataset CIFAR100 --epochs 30 \
    --teacher-layer layer3 --teacher-channels 1024 \
    --student-layers layer2 layer3 layer4 fc --student-channels 512 \
    --sparsity 1e-4

# SAE Weight Compressor
python train.py --method sae_weightcompress \
    --teacher-model resnet50 --student-model resnet18 \
    --dataset CIFAR100 --epochs 30 \
    --swc-teacher-layers layer3:1024 layer4:2048 \
    --swc-student-layers layer3:256  layer4:512  \
    --sparsity 1e-4
"""

import os
import sys
import csv
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import TeacherModel, StudentModel
from data import get_dataloaders
from utils import set_seed
from recovery_utils import validate_recovery_site

# ─── Per-method imports ──────────────────────────────────────────────────────
from logits_kd import LogitKD
from dkd import DKD
from fitnet import FitNet
from crd import CRD, CRDDataset
from sae_injection import SAEInjection
from sae_weightcompress import SAEWeightCompressor

# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Unified Knowledge Distillation Training Script",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Core training ─────────────────────────────────────────────────────────
    p.add_argument('--method', type=str, required=True,
                   choices=['logit_kd', 'dkd', 'fitnet', 'crd',
                            'sae_injection', 'sae_weightcompress'],
                   help='Distillation method to use.')
    p.add_argument('--dataset', type=str, default='CIFAR100',
                   help='Dataset name (CIFAR10, CIFAR100, IMAGENETTE, FOOD101, CUSTOM).')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--seed', type=int, default=42)

    # ── Models ────────────────────────────────────────────────────────────────
    p.add_argument('--teacher-model', type=str, default='resnet50')
    p.add_argument('--student-model', type=str, default='resnet18')
    p.add_argument('--teacher-weights', type=str, default=None,
                   help='Path to pretrained teacher weights (.pth).')
    p.add_argument('--student-weights', type=str, default=None,
                   help='Path to pretrained student weights (.pth).')

    # ── Loss weights / temperature ────────────────────────────────────────────
    p.add_argument('--temperature', type=float, default=4.0)
    p.add_argument('--cls-weight', type=float, default=1.0,
                   help='Weight for cross-entropy classification loss.')
    p.add_argument('--kd-weight', type=float, default=1.0,
                   help='Weight for the primary distillation loss.')
    p.add_argument('--sae-weight', type=float, default=1.0,
                   help='Weight for SAE reconstruction/sparsity loss (SAE methods).')

    # ── Logging / saving ──────────────────────────────────────────────────────
    p.add_argument('--save-dir', type=str, default='logs')
    p.add_argument('--model-dir', type=str, default='weights')
    p.add_argument('--experiment-name', type=str, default=None,
                   help='Base name for log file and saved weights. '
                        'Defaults to --method if not set.')

    # ── DKD-specific ──────────────────────────────────────────────────────────
    p.add_argument('--dkd-alpha', type=float, default=1.0,
                   help='[DKD] Weight for TCKD (target-class KD) component.')
    p.add_argument('--dkd-beta', type=float, default=1.0,
                   help='[DKD] Weight for NCKD (non-target KD) component.')

    # ── FitNet-specific ───────────────────────────────────────────────────────
    p.add_argument('--teacher-layers', nargs='+', type=str, default=None,
                   help='[FitNet] Teacher layer specs: "layer_name:channels" '
                        '(e.g. layer3:1024). Multiple allowed.')
    p.add_argument('--student-layers', nargs='+', type=str, default=None,
                   help='[FitNet] Student layer specs: "layer_name:channels". '
                        'Must correspond 1-to-1 with --teacher-layers.\n'
                        '[SAEInjection] Plain layer names for student trunk '
                        '(e.g. layer2 layer3 layer4 fc).')
    p.add_argument('--fitnet-adapter', type=str, default='student',
                   choices=['student', 'teacher', 'teacher_SAE', 'student_SAE'],
                   help='[FitNet] Which side gets the adaptation layer.')
    p.add_argument('--sparsity', type=float, default=1e-4,
                   help='[FitNet SAE / SAEInjection / SAEWeightCompressor] '
                        'L1 sparsity coefficient for the SAE.')
    p.add_argument('--hint', type=float, default=0.5,
                   help='[FitNet] Hint loss weight')

    # ── CRD-specific ──────────────────────────────────────────────────────────
    p.add_argument('--crd-teacher-layer', type=str, default=None,
                   help='[CRD] Teacher layer name for feature extraction.')
    p.add_argument('--crd-teacher-channels', type=int, default=None,
                   help='[CRD] Channel count at teacher CRD layer.')
    p.add_argument('--crd-student-layer', type=str, default=None,
                   help='[CRD] Student layer name for feature extraction.')
    p.add_argument('--crd-student-channels', type=int, default=None,
                   help='[CRD] Channel count at student CRD layer.')
    p.add_argument('--crd-feat-dim', type=int, default=128,
                   help='[CRD] Projected feature dimension.')
    p.add_argument('--crd-temperature', type=float, default=0.07,
                   help='[CRD] Contrastive loss temperature.')
    p.add_argument('--crd-momentum', type=float, default=0.5,
                   help='[CRD] Momentum for negative prototype update.')

    # ── SAEInjection-specific ─────────────────────────────────────────────────
    p.add_argument('--teacher-layer', type=str, default=None,
                   help='[SAEInjection] Teacher hint layer name.')
    p.add_argument('--teacher-channels', type=int, default=None,
                   help='[SAEInjection] Channel count at teacher hint layer.')
    p.add_argument('--student-channels', type=int, default=None,
                   help='[SAEInjection] Channel count at first student SAE input.')
    p.add_argument('--freeze-teacher', action='store_true', default=False,
                   help='[SAEInjection] Freeze the teacher trunk parameters.')
    p.add_argument('--recovery-site-check', action='store_true', default=False,
                   help='[SAEInjection] Validate that this split can be recovered to a pure student.')

    # ── SAEWeightCompressor-specific ──────────────────────────────────────────
    p.add_argument('--swc-teacher-layers', nargs='+', type=str, default=None,
                   help='[SAEWeightCompressor] Teacher layer specs: '
                        '"layer_name:channels" (e.g. layer3:1024 layer4:2048).')
    p.add_argument('--swc-student-layers', nargs='+', type=str, default=None,
                   help='[SAEWeightCompressor] Student layer specs: '
                        '"layer_name:channels".')
    p.add_argument('--swc-teacher-weight-layers', nargs='+', type=str, default=None,
                   help='[SAEWeightCompressor] Optional teacher weight layer names.')
    p.add_argument('--swc-student-weight-layers', nargs='+', type=str, default=None,
                   help='[SAEWeightCompressor] Optional student weight layer names.')

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# Helper: parse "layer_name:channels" spec list
# ══════════════════════════════════════════════════════════════════════════════

def parse_layer_specs(specs):
    """
    Convert a list of "name:channels" strings into two parallel lists.
    e.g. ["layer3:1024", "layer4:2048"] -> (["layer3","layer4"], [1024, 2048])
    """
    names, channels = [], []
    for spec in specs:
        parts = spec.split(':')
        if len(parts) != 2:
            raise ValueError(
                f"Layer spec '{spec}' is invalid. Expected format 'layer_name:channels'."
            )
        names.append(parts[0])
        channels.append(int(parts[1]))
    return names, channels


# ══════════════════════════════════════════════════════════════════════════════
# Print experimental setup
# ══════════════════════════════════════════════════════════════════════════════

def print_setup(args, device, num_classes, teacher, student, kd_model):
    W = 80
    print(f"\n{'═'*W}")
    print(f"{'KNOWLEDGE DISTILLATION — EXPERIMENTAL SETUP':^{W}}")
    print(f"{'═'*W}")

    print(f"  Method            : {args.method.upper()}")
    print(f"  Dataset           : {args.dataset}  ({num_classes} classes)")
    print(f"  Device            : {device}")
    print(f"  Seed              : {args.seed}")
    print()
    print(f"  Teacher model     : {args.teacher_model}")
    if args.teacher_weights:
        print(f"  Teacher weights   : {args.teacher_weights}")
    print(f"  Student model     : {args.student_model}")
    if args.student_weights:
        print(f"  Student weights   : {args.student_weights}")
    print()
    print(f"  Epochs            : {args.epochs}")
    print(f"  Batch size        : {args.batch_size}")
    print(f"  Learning rate     : {args.lr}")
    print(f"  Temperature       : {args.temperature}")
    print(f"  Loss weights      : CLS={args.cls_weight}  KD={args.kd_weight}  SAE={args.sae_weight}")

    # Method-specific details
    print()
    if args.method == 'dkd':
        print(f"  DKD alpha (TCKD)  : {args.dkd_alpha}")
        print(f"  DKD beta  (NCKD)  : {args.dkd_beta}")

    elif args.method == 'fitnet':
        print(f"  Adapter side      : {args.fitnet_adapter}")
        print(f"  Sparsity (SAE)    : {args.sparsity}")
        if args.teacher_layers:
            print(f"  Teacher layers    : {args.teacher_layers}")
        if args.student_layers:
            print(f"  Student layers    : {args.student_layers}")

    elif args.method == 'crd':
        print(f"  Teacher CRD layer : {args.crd_teacher_layer} ({args.crd_teacher_channels}ch)")
        print(f"  Student CRD layer : {args.crd_student_layer} ({args.crd_student_channels}ch)")
        print(f"  Feat dim          : {args.crd_feat_dim}")
        print(f"  CRD temperature   : {args.crd_temperature}")
        print(f"  CRD momentum      : {args.crd_momentum}")

    elif args.method == 'sae_injection':
        print(f"  Teacher hint layer: {args.teacher_layer} ({args.teacher_channels}ch)")
        print(f"  Student trunk     : {args.student_layers}")
        print(f"  SAE input channels: {args.student_channels}")
        print(f"  Freeze teacher    : {args.freeze_teacher}")
        print(f"  Recovery site chk : {args.recovery_site_check}")
        print(f"  Sparsity          : {args.sparsity}")

    elif args.method == 'sae_weightcompress':
        print(f"  Teacher layers    : {args.swc_teacher_layers}")
        print(f"  Student layers    : {args.swc_student_layers}")
        print(f"  Sparsity          : {args.sparsity}")

    # Parameter counts
    t_params = sum(p.numel() for p in teacher.parameters()) / 1e6
    s_params = sum(p.numel() for p in student.parameters()) / 1e6
    kd_params = sum(p.numel() for p in kd_model.parameters() if p.requires_grad) / 1e6
    print()
    print(f"  Teacher params    : {t_params:.2f}M  (frozen)")
    print(f"  Student params    : {s_params:.2f}M")
    print(f"  KD module trainable params : {kd_params:.2f}M")

    print(f"\n  Logs → {args.save_dir}/  |  Weights → {args.model_dir}/")
    print(f"{'═'*W}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Build KD model
# ══════════════════════════════════════════════════════════════════════════════

def build_kd_model(args, teacher, student, num_train_samples, device):
    method = args.method

    if method == 'logit_kd':
        return LogitKD(teacher=teacher, student=student).to(device)

    elif method == 'dkd':
        return DKD(
            teacher=teacher,
            student=student,
            alpha=args.dkd_alpha,
            beta=args.dkd_beta,
            temperature=args.temperature,
        ).to(device)

    elif method == 'fitnet':
        if not args.teacher_layers or not args.student_layers:
            raise ValueError("--teacher-layers and --student-layers are required for FitNet.")
        t_names, t_channels = parse_layer_specs(args.teacher_layers)
        s_names, s_channels = parse_layer_specs(args.student_layers)
        teacher_layer_pairs = list(zip(t_names, t_channels))
        student_layer_pairs = list(zip(s_names, s_channels))
        return FitNet(
            teacher=teacher,
            student=student,
            layer_pairs=(teacher_layer_pairs, student_layer_pairs),
            adapter=args.fitnet_adapter,
            sparsity=args.sparsity,
        ).to(device)

    elif method == 'crd':
        for attr in ['crd_teacher_layer', 'crd_teacher_channels',
                     'crd_student_layer', 'crd_student_channels']:
            if getattr(args, attr) is None:
                raise ValueError(f"--{attr.replace('_', '-')} is required for CRD.")
        return CRD(
            teacher=teacher,
            student=student,
            teacher_layer_name=args.crd_teacher_layer,
            student_layer_name=args.crd_student_layer,
            teacher_channels=args.crd_teacher_channels,
            student_channels=args.crd_student_channels,
            n_data=num_train_samples,
            feat_dim=args.crd_feat_dim,
            temperature=args.crd_temperature,
            momentum=args.crd_momentum,
        ).to(device)

    elif method == 'sae_injection':
        for attr in ['teacher_layer', 'teacher_channels',
                     'student_layers', 'student_channels']:
            if getattr(args, attr) is None:
                raise ValueError(f"--{attr.replace('_', '-')} is required for SAEInjection.")
        return SAEInjection(
            teacher=teacher,
            student=student,
            teacher_layer_name=args.teacher_layer,
            teacher_channels=args.teacher_channels,
            student_layer_names=args.student_layers,
            student_channels=args.student_channels,
            sparsity=args.sparsity,
            freeze_teacher=args.freeze_teacher,
        ).to(device)

    elif method == 'sae_weightcompress':
        if not args.swc_teacher_layers or not args.swc_student_layers:
            raise ValueError("--swc-teacher-layers and --swc-student-layers required for SAEWeightCompressor.")
        t_names, t_channels = parse_layer_specs(args.swc_teacher_layers)
        s_names, s_channels = parse_layer_specs(args.swc_student_layers)
        return SAEWeightCompressor(
            teacher=teacher,
            student=student,
            teacher_layer_names=t_names,
            teacher_channels=t_channels,
            student_layer_names=s_names,
            student_channels=s_channels,
            sparsity=args.sparsity,
            teacher_weight_layers=args.swc_teacher_weight_layers,
            student_weight_layers=args.swc_student_weight_layers,
        ).to(device)

    else:
        raise ValueError(f"Unknown method: {method}")


# ══════════════════════════════════════════════════════════════════════════════
# Forward pass dispatcher  — returns (teacher_logits, student_logits, aux_loss)
# ══════════════════════════════════════════════════════════════════════════════

def forward_kd(method, kd_model, data, target, device):
    """
    Unified forward call that handles method-specific signatures.

    Returns:
        teacher_logits (Tensor)
        student_logits (Tensor)
        aux_loss       (Tensor | None)  — method-specific distillation loss
                                          (not including CE classification loss)
    """
    if method == 'sae_injection':
        # SAEInjection returns (student_logits, sae_loss); teacher logits are
        # obtained separately (needed for the KL term in the training loop).
        student_logits, sae_loss = kd_model(data)
        with torch.no_grad():
            teacher_logits = kd_model.teacher(data)
        return teacher_logits, student_logits, sae_loss

    elif method == 'dkd':
        # DKD needs ground-truth labels for the target-class mask.
        return kd_model(data, target)

    elif method == 'fitnet':
        # FitNet returns 4 values: teacher_logits, student_logits, hint_loss, sae_loss
        t_logits, s_logits, hint_loss, sae_loss = kd_model(data)
        # Combine hint + sae losses into a single auxiliary loss
        combined = hint_loss
        if sae_loss is not None:
            combined = combined + sae_loss
        return t_logits, s_logits, combined

    else:
        # logit_kd, crd, sae_weightcompress all return (t_logits, s_logits, aux)
        return kd_model(data)


# ══════════════════════════════════════════════════════════════════════════════
# Loss computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_loss(method, teacher_logits, student_logits, target,
                 aux_loss, args, device):
    """
    Combine classification CE loss with the method-specific distillation loss.

    For most methods the total loss is:
        cls_weight * CE(student, y) + kd_weight * distil_loss

    For SAEInjection an additional SAE regulariser is added:
        ... + sae_weight * sae_loss

    For DKD the KL term is already embedded inside aux_loss (TCKD + NCKD),
    so we skip the generic KL term.

    Returns:
        loss          (Tensor): scalar total loss
        cls_loss_val  (float)
        kd_loss_val   (float)
        aux_loss_val  (float)
    """
    ce_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    T = args.temperature

    cls_loss = ce_criterion(student_logits, target)
    cls_loss_val = cls_loss.item()

    # ── KL distillation term (used by logit_kd, crd, sae_injection, fitnet, swc) ──
    kl_loss = kl_criterion(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
    ) * (T ** 2)

    zero = torch.tensor(0.0, device=device)
    aux = aux_loss if aux_loss is not None else zero

    if method == 'dkd':
        # aux_loss IS the full DKD loss (no separate KL term)
        total = args.cls_weight * cls_loss + args.kd_weight * aux
        return total, cls_loss_val, aux.item(), 0.0

    elif method == 'sae_injection':
        # aux_loss = sae_loss; KL is the distillation signal
        total = (args.cls_weight * cls_loss
                 + args.kd_weight  * kl_loss
                 + args.sae_weight * aux)
        return total, cls_loss_val, kl_loss.item(), aux.item()

    elif method == 'fitnet':
        # aux_loss = hint_loss (+ sae_loss if applicable); keep KL as well
        total = (args.cls_weight * cls_loss
                 + args.kd_weight  * kl_loss
                 + args.sae_weight * aux)
        return total, cls_loss_val, kl_loss.item(), aux.item()

    elif method == 'sae_weightcompress':
        # aux_loss = sae_loss; KL is the distillation signal
        total = (args.cls_weight * cls_loss
                 + args.kd_weight  * kl_loss
                 + args.sae_weight * aux)
        return total, cls_loss_val, kl_loss.item(), aux.item()

    else:
        # logit_kd uses KL directly; CRD uses its own auxiliary criterion.
        if method == 'logit_kd' or aux_loss is None:
            kd_loss = kl_loss
        else:
            kd_loss = aux
        total = args.cls_weight * cls_loss + args.kd_weight * kd_loss
        return total, cls_loss_val, kd_loss.item(), 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Train one epoch
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(kd_model, optimizer, dataloader, device, args, epoch):
    kd_model.train()
    # Ensure teacher stays in eval mode inside the KD wrapper
    if hasattr(kd_model, 'teacher'):
        kd_model.teacher.eval()

    totals = dict(loss=0., cls=0., kd=0., aux=0., correct=0, samples=0)
    num_batches = len(dataloader)
    print(f"\nEpoch {epoch}/{args.epochs} [TRAIN] — {num_batches} batches", flush=True)

    for batch_idx, batch in enumerate(dataloader):
        # CRD wraps dataset to emit (data, target, idx)
        if args.method == 'crd' and len(batch) == 3:
            data, target, sample_idx = batch
            data, target = data.to(device), target.to(device)
            kd_model.set_sample_indices(sample_idx.to(device))
        else:
            data, target = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        t_logits, s_logits, aux_loss = forward_kd(
            args.method, kd_model, data, target, device)

        loss, cls_v, kd_v, aux_v = compute_loss(
            args.method, t_logits, s_logits, target, aux_loss, args, device)

        loss.backward()
        optimizer.step()

        bs = data.size(0)
        totals['loss']    += loss.item() * bs
        totals['cls']     += cls_v       * bs
        totals['kd']      += kd_v        * bs
        totals['aux']     += aux_v       * bs
        totals['correct'] += (s_logits.argmax(1) == target).sum().item()
        totals['samples'] += bs

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            acc = 100. * totals['correct'] / totals['samples']
            print(f"  [{batch_idx+1:4d}/{num_batches}] "
                  f"Loss={loss.item():.4f}  CLS={cls_v:.4f}  "
                  f"KD={kd_v:.4f}  AUX={aux_v:.6f}  Acc={acc:.2f}%",
                  flush=True)

    n = totals['samples']
    return (totals['loss']/n, totals['cls']/n,
            totals['kd']/n,  totals['aux']/n,
            totals['correct']/n)


# ══════════════════════════════════════════════════════════════════════════════
# Evaluate one epoch
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_one_epoch(kd_model, dataloader, device, args, epoch):
    kd_model.eval()

    totals = dict(loss=0., cls=0., kd=0., aux=0., correct=0, samples=0)
    num_batches = len(dataloader)
    print(f"\nEpoch {epoch}/{args.epochs} [EVAL]  — {num_batches} batches", flush=True)

    for batch_idx, batch in enumerate(dataloader):
        if args.method == 'crd' and len(batch) == 3:
            data, target, sample_idx = batch
            data, target = data.to(device), target.to(device)
            kd_model.set_sample_indices(sample_idx.to(device))
        else:
            data, target = batch[0].to(device), batch[1].to(device)

        t_logits, s_logits, aux_loss = forward_kd(
            args.method, kd_model, data, target, device)

        loss, cls_v, kd_v, aux_v = compute_loss(
            args.method, t_logits, s_logits, target, aux_loss, args, device)

        bs = data.size(0)
        totals['loss']    += loss.item() * bs
        totals['cls']     += cls_v       * bs
        totals['kd']      += kd_v        * bs
        totals['aux']     += aux_v       * bs
        totals['correct'] += (s_logits.argmax(1) == target).sum().item()
        totals['samples'] += bs

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            acc = 100. * totals['correct'] / totals['samples']
            print(f"  [{batch_idx+1:4d}/{num_batches}] "
                  f"Loss={loss.item():.4f}  CLS={cls_v:.4f}  "
                  f"KD={kd_v:.4f}  AUX={aux_v:.6f}  Acc={acc:.2f}%",
                  flush=True)

    n = totals['samples']
    return (totals['loss']/n, totals['cls']/n,
            totals['kd']/n,  totals['aux']/n,
            totals['correct']/n)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.experiment_name is None:
        args.experiment_name = args.method

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, eval_loader, num_classes = get_dataloaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
    )

    # CRD needs sample-index tracking — wrap the underlying dataset
    if args.method == 'crd':
        from crd import CRDDataset
        from torch.utils.data import DataLoader
        crd_train_set = CRDDataset(train_loader.dataset)
        train_loader = DataLoader(
            crd_train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    num_train = len(train_loader.dataset)

    # ── Models ────────────────────────────────────────────────────────────────
    teacher = TeacherModel(
        model_name=args.teacher_model,
        num_classes=num_classes,
        weights_path=args.teacher_weights,
        pretrained=True,
    ).to(device)

    student = StudentModel(
        model_name=args.student_model,
        num_classes=num_classes,
        weights_path=args.student_weights,
    ).to(device)

    # Freeze teacher globally (each KD wrapper may handle this internally too)
    for param in teacher.parameters():
        param.requires_grad = False
    teacher.eval()

    # ── KD Model ──────────────────────────────────────────────────────────────
    if args.method == 'sae_injection' and args.recovery_site_check:
        site_check = validate_recovery_site(
            teacher_model=teacher.model,
            student_model=student.model,
            teacher_layer_name=args.teacher_layer,
            student_layer_names=args.student_layers,
            teacher_channels=args.teacher_channels,
            student_channels=args.student_channels,
            device=device,
        )
        print("\n[Recovery Site Check]")
        print(f"  Recoverable       : {site_check.recoverable}")
        print(f"  Reason            : {site_check.reason}")
        print(f"  Source params     : {site_check.source_param_count:,}")
        print(f"  Target params     : {site_check.target_param_count:,}")
        print(f"  Teacher prefix    : {site_check.shape_teacher_prefix}")
        print(f"  Encoded trunk     : {site_check.shape_teacher_trunk_encoded}")
        print(f"  Discarded prefix  : {site_check.shape_discarded_student_prefix}")
        if site_check.error:
            print(f"  Probe error       : {site_check.error}")
        if not site_check.recoverable:
            raise ValueError(
                "Recovery site check failed. "
                f"reason={site_check.reason}, "
                f"source_params={site_check.source_param_count}, "
                f"target_params={site_check.target_param_count}, "
                f"encoded_shape={site_check.shape_teacher_trunk_encoded}, "
                f"discarded_shape={site_check.shape_discarded_student_prefix}"
            )

    kd_model = build_kd_model(args, teacher, student, num_train, device)

    # ── Print setup ───────────────────────────────────────────────────────────
    print_setup(args, device, num_classes, teacher, student, kd_model)

    # ── Optimizer — only student (+ KD adapter) params ───────────────────────
    if args.method == 'sae_injection':
        # SAEInjection exposes its trainable params via .hybrid_model
        trainable = kd_model.hybrid_model.parameters()
    else:
        trainable = (p for p in kd_model.parameters() if p.requires_grad)

    optimizer = optim.Adam(trainable, lr=args.lr)

    # ── Directories & CSV ─────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, f'{args.experiment_name}.csv')

    best_acc   = 0.0
    best_epoch = 0
    best_model_path = os.path.join(args.model_dir, f'{args.experiment_name}_best.pth')

    CSV_HEADER = [
        'epoch',
        'train_loss', 'train_cls_loss', 'train_kd_loss', 'train_aux_loss', 'train_acc',
        'val_loss',   'val_cls_loss',   'val_kd_loss',   'val_aux_loss',   'val_acc',
        'lr', 'epoch_time_s',
    ]

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()

            # Train
            tr_loss, tr_cls, tr_kd, tr_aux, tr_acc = train_one_epoch(
                kd_model, optimizer, train_loader, device, args, epoch)

            # Eval
            vl_loss, vl_cls, vl_kd, vl_aux, vl_acc = eval_one_epoch(
                kd_model, eval_loader, device, args, epoch)

            elapsed = time.perf_counter() - t0
            lr_now  = optimizer.param_groups[0]['lr']

            # ── Terminal summary ───────────────────────────────────────────
            W = 80
            print(f"\n{'═'*W}", flush=True)
            print(f"EPOCH {epoch}/{args.epochs}  ({elapsed:.1f}s)  LR={lr_now:.2e}", flush=True)
            print(f"{'─'*W}", flush=True)
            print(f"  TRAIN  Loss={tr_loss:.4f}  CLS={tr_cls:.4f}  "
                  f"KD={tr_kd:.4f}  AUX={tr_aux:.6f}  Acc={tr_acc*100:.2f}%", flush=True)
            print(f"  EVAL   Loss={vl_loss:.4f}  CLS={vl_cls:.4f}  "
                  f"KD={vl_kd:.4f}  AUX={vl_aux:.6f}  Acc={vl_acc*100:.2f}%", flush=True)
            print(f"{'─'*W}", flush=True)

            if vl_acc > best_acc:
                best_acc   = vl_acc
                best_epoch = epoch
                torch.save(kd_model.state_dict(), best_model_path)
                print(f"  ★ New best: {best_acc*100:.2f}%  → saved to {best_model_path}", flush=True)
            else:
                print(f"  Best so far: {best_acc*100:.2f}%  (epoch {best_epoch})", flush=True)

            print(f"{'═'*W}\n", flush=True)

            # ── CSV row ───────────────────────────────────────────────────
            writer.writerow([
                epoch,
                tr_loss, tr_cls, tr_kd, tr_aux, tr_acc,
                vl_loss, vl_cls, vl_kd, vl_aux, vl_acc,
                lr_now, elapsed,
            ])
            f.flush()

    # ── Save final weights ────────────────────────────────────────────────────
    final_path = os.path.join(args.model_dir, f'{args.experiment_name}_final.pth')
    torch.save(kd_model.state_dict(), final_path)

    W = 80
    print(f"\n{'═'*W}")
    print(f"{'TRAINING COMPLETE':^{W}}")
    print(f"{'═'*W}")
    print(f"  Best val accuracy : {best_acc*100:.2f}%  (epoch {best_epoch})")
    print(f"  Best weights      : {best_model_path}")
    print(f"  Final weights     : {final_path}")
    print(f"  Training log      : {csv_path}")
    print(f"{'═'*W}\n")


if __name__ == '__main__':
    main()
