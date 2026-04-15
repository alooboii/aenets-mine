import argparse
import csv
import os
from typing import List

import torch

from data import get_dataloaders
from models import StudentModel, TeacherModel
from recovery_utils import (
    RESNET_STAGE_ORDER,
    RecoverySiteCheckResult,
    build_discarded_student_prefix,
    build_teacher_prefix,
    count_parameters,
    infer_student_layers_from_boundary,
    parse_stage_name,
    validate_recovery_site,
)
from utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Precheck candidate AENets injection sites for student recovery compatibility."
    )
    p.add_argument("--teacher-model", type=str, default="resnet50")
    p.add_argument("--student-model", type=str, default="resnet18")
    p.add_argument("--dataset", type=str, default="CIFAR100")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--teacher-layers", nargs="+", type=str, default=None)
    p.add_argument("--student-boundaries", nargs="+", type=str, default=None)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--probe-batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sparsity", type=float, default=1e-4)
    p.add_argument("--save-dir", type=str, default="logs")
    p.add_argument("--experiment-name", type=str, default="aenets_precheck")
    p.add_argument("--default-epochs", type=int, default=30)
    p.add_argument("--default-recover-epochs", type=int, default=20)
    p.add_argument("--default-recover-lr", type=float, default=1e-3)
    return p.parse_args()


def _shape_from_prefix(prefix: torch.nn.Module, device: torch.device, image_size: int, batch_size: int):
    prefix = prefix.to(device).eval()
    with torch.no_grad():
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        out = prefix(x)
    return tuple(out.shape)


def _format_train_cmd(args, teacher_layer: str, teacher_channels: int, student_layers: List[str], student_channels: int, run_name: str) -> str:
    student_layers_joined = " ".join(student_layers)
    return (
        f"python train.py --method sae_injection "
        f"--teacher-model {args.teacher_model} --student-model {args.student_model} "
        f"--dataset {args.dataset} --batch-size {args.batch_size} --epochs {args.default_epochs} "
        f"--teacher-layer {teacher_layer} --teacher-channels {teacher_channels} "
        f"--student-layers {student_layers_joined} --student-channels {student_channels} "
        f"--sparsity {args.sparsity} --recovery-site-check "
        f"--experiment-name {run_name}"
    )


def _format_recover_cmd(args, teacher_layer: str, teacher_channels: int, student_layers: List[str], student_channels: int, run_name: str) -> str:
    student_layers_joined = " ".join(student_layers)
    return (
        f"python recover_student_prefix.py "
        f"--aenets-checkpoint weights/{run_name}_best.pth "
        f"--teacher-model {args.teacher_model} --student-model {args.student_model} "
        f"--dataset {args.dataset} --batch-size {args.batch_size} "
        f"--teacher-layer {teacher_layer} --teacher-channels {teacher_channels} "
        f"--student-layers {student_layers_joined} --student-channels {student_channels} "
        f"--sparsity {args.sparsity} "
        f"--recover-epochs {args.default_recover_epochs} --recover-lr {args.default_recover_lr} "
        f"--experiment-name {run_name}_recover"
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, _, num_classes = get_dataloaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        root=args.data_root,
    )
    print(f"Dataset: {args.dataset} | train samples: {len(train_loader.dataset)} | classes: {num_classes}")
    print(f"Device: {device}")

    teacher = TeacherModel(
        model_name=args.teacher_model,
        num_classes=num_classes,
        weights_path=None,
        pretrained=True,
    ).to(device)
    student = StudentModel(
        model_name=args.student_model,
        num_classes=num_classes,
        weights_path=None,
    ).to(device)

    teacher_layers = args.teacher_layers or list(RESNET_STAGE_ORDER)
    student_boundaries = args.student_boundaries or list(RESNET_STAGE_ORDER)
    teacher_layers = [parse_stage_name(s, strict_stage_level=True) for s in teacher_layers]
    student_boundaries = [parse_stage_name(s, strict_stage_level=True) for s in student_boundaries]

    rows = []
    candidate_id = 0

    for boundary in student_boundaries:
        student_layers = infer_student_layers_from_boundary(boundary)
        discarded_prefix = build_discarded_student_prefix(student.model, boundary)
        discarded_shape = _shape_from_prefix(
            discarded_prefix, device=device, image_size=args.image_size, batch_size=args.probe_batch_size
        )
        student_channels = int(discarded_shape[1])
        target_params = count_parameters(discarded_prefix)

        for teacher_stage in teacher_layers:
            candidate_id += 1
            teacher_prefix = build_teacher_prefix(teacher.model, teacher_stage)
            teacher_shape = _shape_from_prefix(
                teacher_prefix, device=device, image_size=args.image_size, batch_size=args.probe_batch_size
            )
            teacher_channels = int(teacher_shape[1])

            result: RecoverySiteCheckResult = validate_recovery_site(
                teacher_model=teacher.model,
                student_model=student.model,
                teacher_layer_name=teacher_stage,
                student_layer_names=student_layers,
                teacher_channels=teacher_channels,
                student_channels=student_channels,
                device=device,
                image_size=args.image_size,
                batch_size=args.probe_batch_size,
            )

            run_name = f"{args.experiment_name}_{teacher_stage}_to_{boundary}"
            train_cmd = _format_train_cmd(
                args=args,
                teacher_layer=teacher_stage,
                teacher_channels=teacher_channels,
                student_layers=student_layers,
                student_channels=student_channels,
                run_name=run_name,
            )
            recover_cmd = _format_recover_cmd(
                args=args,
                teacher_layer=teacher_stage,
                teacher_channels=teacher_channels,
                student_layers=student_layers,
                student_channels=student_channels,
                run_name=run_name,
            )

            rows.append(
                {
                    "candidate_id": candidate_id,
                    "teacher_layer": teacher_stage,
                    "student_boundary": boundary,
                    "student_layers": " ".join(student_layers),
                    "teacher_channels": teacher_channels,
                    "student_channels": student_channels,
                    "source_param_count": result.source_param_count,
                    "target_param_count": target_params,
                    "param_margin": result.source_param_count - target_params,
                    "shape_teacher_prefix": str(result.shape_teacher_prefix),
                    "shape_teacher_trunk_encoded": str(result.shape_teacher_trunk_encoded),
                    "shape_discarded_student_prefix": str(result.shape_discarded_student_prefix),
                    "recoverable": int(result.recoverable),
                    "reason": result.reason,
                    "error": result.error or "",
                    "train_cmd": train_cmd,
                    "recover_cmd": recover_cmd,
                }
            )

    rows.sort(
        key=lambda r: (
            -r["recoverable"],
            -(r["param_margin"]),
            r["teacher_layer"],
            r["student_boundary"],
        )
    )

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, f"{args.experiment_name}.csv")
    header = [
        "candidate_id",
        "teacher_layer",
        "student_boundary",
        "student_layers",
        "teacher_channels",
        "student_channels",
        "source_param_count",
        "target_param_count",
        "param_margin",
        "shape_teacher_prefix",
        "shape_teacher_trunk_encoded",
        "shape_discarded_student_prefix",
        "recoverable",
        "reason",
        "error",
        "train_cmd",
        "recover_cmd",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Ranked Candidate Summary ===")
    for row in rows:
        status = "OK" if row["recoverable"] else f"FAIL({row['reason']})"
        print(
            f"[{row['candidate_id']:02d}] {status} "
            f"T:{row['teacher_layer']} -> S:{row['student_boundary']} | "
            f"margin={row['param_margin']:,} | "
            f"enc={row['shape_teacher_trunk_encoded']} vs disc={row['shape_discarded_student_prefix']}"
        )

    print(f"\nSaved precheck CSV: {csv_path}")
    ok_rows = [r for r in rows if r["recoverable"] == 1]
    if ok_rows:
        top = ok_rows[0]
        print("\nTop recoverable site commands:")
        print(f"  TRAIN  : {top['train_cmd']}")
        print(f"  RECOVER: {top['recover_cmd']}")
    else:
        print("\nNo recoverable site found. Try different candidate layers/boundaries.")


if __name__ == "__main__":
    main()
