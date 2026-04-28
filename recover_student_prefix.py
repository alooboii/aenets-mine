import argparse
import copy
import csv
import json
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import get_dataloaders
from models import StudentModel, TeacherModel
from recovery_utils import (
    build_discarded_student_prefix,
    build_retained_student_tail,
    copy_prefix_weights,
    family_depth_prune_and_pack,
    freeze_module,
    resolve_student_boundary,
    topk_prune_and_pack,
    unfreeze_module,
    validate_recovery_site,
)
from sae_injection import SAEInjection
from utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Recover discarded student prefix from a trained SAEInjection checkpoint."
    )
    p.add_argument("--aenets-checkpoint", type=str, required=True)

    p.add_argument("--teacher-model", type=str, default="resnet50")
    p.add_argument("--student-model", type=str, default="resnet18")
    p.add_argument("--teacher-weights", type=str, default=None)
    p.add_argument("--student-weights", type=str, default=None)

    p.add_argument("--dataset", type=str, default="CIFAR100")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=64)

    p.add_argument("--teacher-layer", type=str, required=True)
    p.add_argument("--teacher-channels", type=int, required=True)
    p.add_argument("--student-layers", nargs="+", type=str, required=True)
    p.add_argument("--student-channels", type=int, required=True)
    p.add_argument("--sparsity", type=float, default=1e-4)
    p.add_argument("--freeze-teacher", action="store_true", default=False)

    p.add_argument("--recover-epochs", type=int, default=20)
    p.add_argument("--recover-lr", type=float, default=1e-3)
    p.add_argument("--recover-mse-weight", type=float, default=1.0)
    p.add_argument("--recover-kd-weight", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument(
        "--recover-init-method",
        type=str,
        choices=["global_topk", "family_depth_stage"],
        default="global_topk",
        help="Recovered prefix initialization policy.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--track-assembled-val-acc", action="store_true", default=False)

    p.add_argument("--save-dir", type=str, default="logs")
    p.add_argument("--model-dir", type=str, default="weights")
    p.add_argument("--experiment-name", type=str, default="sae_recovery")
    return p.parse_args()


def _safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def eval_student_accuracy(student_wrapper: StudentModel, dataloader, device: torch.device) -> float:
    student_wrapper.eval()
    total = 0
    correct = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        logits = student_wrapper(data)
        pred = logits.argmax(dim=1)
        total += target.size(0)
        correct += (pred == target).sum().item()
    return correct / max(total, 1)


@torch.no_grad()
def eval_hybrid_accuracy(kd_model: SAEInjection, dataloader, device: torch.device) -> float:
    kd_model.eval()
    total = 0
    correct = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        student_logits, _ = kd_model(data)
        pred = student_logits.argmax(dim=1)
        total += target.size(0)
        correct += (pred == target).sum().item()
    return correct / max(total, 1)


def train_prefix_epoch(
    recovered_prefix: nn.Module,
    teacher_prefix: nn.Module,
    sae_encoder: nn.Module,
    trained_tail: nn.Module,
    teacher_model: nn.Module,
    optimizer: optim.Optimizer,
    dataloader,
    device: torch.device,
    mse_weight: float,
    kd_weight: float,
    temperature: float,
) -> Dict[str, float]:
    recovered_prefix.train()
    teacher_prefix.eval()
    sae_encoder.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_kd = 0.0
    total_samples = 0

    for data, _ in dataloader:
        data = data.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            target = sae_encoder(teacher_prefix(data))
            teacher_logits = teacher_model(data) if kd_weight > 0.0 else None

        pred = recovered_prefix(data)
        mse_loss = F.mse_loss(pred, target)

        if kd_weight > 0.0:
            student_logits = trained_tail(pred)
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature ** 2)
        else:
            kd_loss = torch.tensor(0.0, device=device)

        loss = mse_weight * mse_loss + kd_weight * kd_loss
        loss.backward()
        optimizer.step()

        bs = data.size(0)
        total_loss += loss.item() * bs
        total_mse += mse_loss.item() * bs
        total_kd += kd_loss.item() * bs
        total_samples += bs

    n = max(total_samples, 1)
    return {
        "total": total_loss / n,
        "mse": total_mse / n,
        "kd": total_kd / n,
    }


@torch.no_grad()
def eval_prefix_epoch(
    recovered_prefix: nn.Module,
    teacher_prefix: nn.Module,
    sae_encoder: nn.Module,
    trained_tail: nn.Module,
    teacher_model: nn.Module,
    dataloader,
    device: torch.device,
    mse_weight: float,
    kd_weight: float,
    temperature: float,
) -> Dict[str, float]:
    recovered_prefix.eval()
    teacher_prefix.eval()
    sae_encoder.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_kd = 0.0
    total_samples = 0

    for data, _ in dataloader:
        data = data.to(device)
        target = sae_encoder(teacher_prefix(data))
        pred = recovered_prefix(data)
        mse_loss = F.mse_loss(pred, target)
        if kd_weight > 0.0:
            teacher_logits = teacher_model(data)
            student_logits = trained_tail(pred)
            kd_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(teacher_logits / temperature, dim=1),
                reduction="batchmean",
            ) * (temperature ** 2)
        else:
            kd_loss = torch.tensor(0.0, device=device)

        loss = mse_weight * mse_loss + kd_weight * kd_loss

        bs = data.size(0)
        total_loss += loss.item() * bs
        total_mse += mse_loss.item() * bs
        total_kd += kd_loss.item() * bs
        total_samples += bs

    n = max(total_samples, 1)
    return {
        "total": total_loss / n,
        "mse": total_mse / n,
        "kd": total_kd / n,
    }


def main():
    args = parse_args()
    if args.recover_mse_weight < 0.0 or args.recover_kd_weight < 0.0:
        raise ValueError("--recover-mse-weight and --recover-kd-weight must be non-negative.")
    if args.recover_mse_weight == 0.0 and args.recover_kd_weight == 0.0:
        raise ValueError("At least one of --recover-mse-weight or --recover-kd-weight must be > 0.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    train_loader, eval_loader, num_classes = get_dataloaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        root=args.data_root,
    )

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

    kd_model = SAEInjection(
        teacher=teacher,
        student=student,
        teacher_layer_name=args.teacher_layer,
        teacher_channels=args.teacher_channels,
        student_layer_names=args.student_layers,
        student_channels=args.student_channels,
        sparsity=args.sparsity,
        freeze_teacher=args.freeze_teacher,
    ).to(device)

    checkpoint = _safe_torch_load(args.aenets_checkpoint, device)
    kd_model.load_state_dict(checkpoint, strict=True)
    if args.teacher_weights:
        # Ensure logit-KD uses the explicitly requested teacher checkpoint,
        # even when the hybrid checkpoint also contains teacher parameters.
        kd_model.teacher.load_teacher_weights(args.teacher_weights)

    site_check = validate_recovery_site(
        teacher_model=kd_model.teacher.model,
        student_model=kd_model.student.model,
        teacher_layer_name=args.teacher_layer,
        student_layer_names=args.student_layers,
        teacher_channels=args.teacher_channels,
        student_channels=args.student_channels,
        device=device,
    )
    if not site_check.recoverable:
        raise ValueError(
            "Recovery site is not valid for prefix recovery. "
            f"reason={site_check.reason}, "
            f"source_params={site_check.source_param_count}, "
            f"target_params={site_check.target_param_count}, "
            f"encoded_shape={site_check.shape_teacher_trunk_encoded}, "
            f"discarded_shape={site_check.shape_discarded_student_prefix}"
        )

    boundary_stage = resolve_student_boundary(args.student_layers, strict_stage_level=True)

    teacher_prefix = kd_model.hybrid_model.teacher_trunk.to(device)
    sae_encoder = kd_model.hybrid_model.sae_adapter.encoder.to(device)
    trained_tail = build_retained_student_tail(kd_model.student.model, boundary_stage).to(device)
    discarded_prefix_template = build_discarded_student_prefix(
        kd_model.student.model, boundary_stage
    )
    recovered_prefix = copy.deepcopy(discarded_prefix_template).to(device)

    freeze_module(kd_model.teacher)
    freeze_module(teacher_prefix)
    freeze_module(sae_encoder)
    freeze_module(trained_tail)
    unfreeze_module(recovered_prefix)

    frozen_groups = {
        "teacher": kd_model.teacher,
        "teacher_prefix": teacher_prefix,
        "sae_encoder": sae_encoder,
        "trained_tail": trained_tail,
    }
    for group_name, module in frozen_groups.items():
        leaked = [n for n, p in module.named_parameters() if p.requires_grad]
        if leaked:
            raise RuntimeError(
                f"Freeze invariant violated: {group_name} has trainable params (e.g. {leaked[:3]})."
            )
    recovered_trainable = [n for n, p in recovered_prefix.named_parameters() if p.requires_grad]
    if not recovered_trainable:
        raise RuntimeError("Recovered prefix has no trainable parameters.")

    if args.recover_init_method == "global_topk":
        prune_stats = topk_prune_and_pack(
            source_modules=[("teacher_prefix", teacher_prefix), ("sae_encoder", sae_encoder)],
            target_module=recovered_prefix,
            device=device,
        )
    elif args.recover_init_method == "family_depth_stage":
        prune_stats = family_depth_prune_and_pack(
            source_modules=[("teacher_prefix", teacher_prefix), ("sae_encoder", sae_encoder)],
            target_module=recovered_prefix,
            device=device,
        )
    else:
        raise ValueError(f"Unknown --recover-init-method: {args.recover_init_method}")

    print(f"Recovery init method      : {args.recover_init_method}")

    optimizer = optim.Adam(recovered_prefix.parameters(), lr=args.recover_lr)

    best_val = float("inf")
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_recovered_val_acc: Optional[float] = None

    csv_path = os.path.join(args.save_dir, f"{args.experiment_name}.csv")
    prefix_best_path = os.path.join(
        args.model_dir, f"{args.experiment_name}_recovered_prefix_best.pth"
    )
    pure_student_wrapper_path = os.path.join(
        args.model_dir, f"{args.experiment_name}_pure_student_wrapper.pth"
    )
    pure_student_model_path = os.path.join(
        args.model_dir, f"{args.experiment_name}_pure_student_model.pth"
    )
    summary_path = os.path.join(args.save_dir, f"{args.experiment_name}_summary.json")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_total_loss",
                "train_mse",
                "train_kd",
                "val_total_loss",
                "val_mse",
                "val_kd",
                "best_val_total",
                "assembled_val_acc",
                "pure_recovered_student_val_acc",
                "epoch_time_s",
            ]
        )

        for epoch in range(1, args.recover_epochs + 1):
            t0 = time.perf_counter()
            train_metrics = train_prefix_epoch(
                recovered_prefix=recovered_prefix,
                teacher_prefix=teacher_prefix,
                sae_encoder=sae_encoder,
                trained_tail=trained_tail,
                teacher_model=kd_model.teacher,
                optimizer=optimizer,
                dataloader=train_loader,
                device=device,
                mse_weight=args.recover_mse_weight,
                kd_weight=args.recover_kd_weight,
                temperature=args.temperature,
            )
            val_metrics = eval_prefix_epoch(
                recovered_prefix=recovered_prefix,
                teacher_prefix=teacher_prefix,
                sae_encoder=sae_encoder,
                trained_tail=trained_tail,
                teacher_model=kd_model.teacher,
                dataloader=eval_loader,
                device=device,
                mse_weight=args.recover_mse_weight,
                kd_weight=args.recover_kd_weight,
                temperature=args.temperature,
            )

            assembled_val_acc: Optional[float] = None
            if args.track_assembled_val_acc:
                temp_student = copy.deepcopy(kd_model.student).to(device)
                copy_prefix_weights(
                    target_student_model=temp_student.model,
                    recovered_prefix=recovered_prefix,
                    boundary_stage=boundary_stage,
                )
                assembled_val_acc = eval_student_accuracy(temp_student, eval_loader, device)
                if best_recovered_val_acc is None or assembled_val_acc > best_recovered_val_acc:
                    best_recovered_val_acc = assembled_val_acc

            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                best_epoch = epoch
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in recovered_prefix.state_dict().items()
                }
                torch.save(best_state, prefix_best_path)

            elapsed = time.perf_counter() - t0
            writer.writerow(
                [
                    epoch,
                    train_metrics["total"],
                    train_metrics["mse"],
                    train_metrics["kd"],
                    val_metrics["total"],
                    val_metrics["mse"],
                    val_metrics["kd"],
                    best_val,
                    assembled_val_acc if assembled_val_acc is not None else "",
                    assembled_val_acc if assembled_val_acc is not None else "",
                    elapsed,
                ]
            )
            f.flush()

            msg = (
                f"Epoch {epoch}/{args.recover_epochs} | "
                f"train_total={train_metrics['total']:.6f} | "
                f"train_mse={train_metrics['mse']:.6f} | "
                f"train_kd={train_metrics['kd']:.6f} | "
                f"val_total={val_metrics['total']:.6f} | "
                f"val_mse={val_metrics['mse']:.6f} | "
                f"val_kd={val_metrics['kd']:.6f} | "
                f"best_val_total={best_val:.6f}"
            )
            if assembled_val_acc is not None:
                msg += f" | pure_recovered_val_acc={assembled_val_acc*100:.2f}%"
            print(msg)

    if best_state is None:
        raise RuntimeError("Recovery training did not produce a best checkpoint.")

    recovered_prefix.load_state_dict(best_state, strict=True)

    pure_student = copy.deepcopy(kd_model.student).to(device)
    copy_prefix_weights(
        target_student_model=pure_student.model,
        recovered_prefix=recovered_prefix,
        boundary_stage=boundary_stage,
    )

    hybrid_val_acc = eval_hybrid_accuracy(kd_model, eval_loader, device)
    pure_student_val_acc = eval_student_accuracy(pure_student, eval_loader, device)

    torch.save(pure_student.state_dict(), pure_student_wrapper_path)
    torch.save(pure_student.model.state_dict(), pure_student_model_path)

    summary = {
        "experiment_name": args.experiment_name,
        "checkpoint": args.aenets_checkpoint,
        "teacher_layer": args.teacher_layer,
        "student_layers": args.student_layers,
        "boundary_stage": boundary_stage,
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "recover_mse_weight": args.recover_mse_weight,
        "recover_kd_weight": args.recover_kd_weight,
        "temperature": args.temperature,
        "recover_init_method": args.recover_init_method,
        "hybrid_val_acc": hybrid_val_acc,
        "pure_student_val_acc": pure_student_val_acc,
        "best_recovered_student_val_acc": best_recovered_val_acc,
        "site_check": {
            "reason": site_check.reason,
            "recoverable": site_check.recoverable,
            "source_param_count": site_check.source_param_count,
            "target_param_count": site_check.target_param_count,
            "shape_teacher_prefix": site_check.shape_teacher_prefix,
            "shape_teacher_trunk_encoded": site_check.shape_teacher_trunk_encoded,
            "shape_discarded_student_prefix": site_check.shape_discarded_student_prefix,
        },
        "prune_stats": prune_stats,
        "artifacts": {
            "training_csv": csv_path,
            "recovered_prefix_best": prefix_best_path,
            "pure_student_wrapper": pure_student_wrapper_path,
            "pure_student_model": pure_student_model_path,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nRecovery complete.")
    print(f"  Best epoch             : {best_epoch}")
    print(f"  Best val total         : {best_val:.6f}")
    print(f"  Hybrid val accuracy    : {hybrid_val_acc*100:.2f}%")
    print(f"  Pure student val acc   : {pure_student_val_acc*100:.2f}%")
    if best_recovered_val_acc is not None:
        print(f"  Best recovered val acc : {best_recovered_val_acc*100:.2f}%")
    print(f"  CSV log                : {csv_path}")
    print(f"  Recovered prefix (best): {prefix_best_path}")
    print(f"  Pure student wrapper   : {pure_student_wrapper_path}")
    print(f"  Pure student model     : {pure_student_model_path}")
    print(f"  Summary JSON           : {summary_path}")


if __name__ == "__main__":
    main()
