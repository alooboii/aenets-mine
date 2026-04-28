import argparse
import csv
import os
from typing import List

import torch

from boomerang_kd import BoomerangKD
from boomerang_utils import count_inference_params, infer_model_family
from data import get_dataloaders
from models import StudentModel, TeacherModel
from utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Boomerang zero-shot interpolated ViT models (K=0..M)."
    )
    p.add_argument("--boomerang-checkpoint", type=str, required=True)

    p.add_argument("--teacher-model", type=str, default="vit_b_16")
    p.add_argument("--student-model", type=str, default="vit_b_16")
    p.add_argument("--teacher-weights", type=str, default=None)
    p.add_argument("--student-weights", type=str, default=None)

    p.add_argument("--dataset", type=str, default="IMAGENETTE")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=64)

    p.add_argument("--model-family", type=str, default="auto", choices=["auto", "cnn", "vit"])
    p.add_argument("--student-num-layers", type=int, required=True)

    p.add_argument("--boomerang-keep-every", type=int, default=2)
    p.add_argument(
        "--boomerang-keep-last-layer",
        type=lambda x: str(x).lower() in {"1", "true", "yes", "y"},
        nargs="?",
        const=True,
        default=True,
    )
    p.add_argument(
        "--boomerang-patch-order",
        type=str,
        default="reverse",
        choices=["reverse", "forward"],
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="logs")
    p.add_argument("--experiment-name", type=str, default="boomerang_interpolation")

    return p.parse_args()


def resolve_model_family(args) -> str:
    if args.model_family != "auto":
        return args.model_family
    t_family = infer_model_family(args.teacher_model)
    s_family = infer_model_family(args.student_model)
    if "vit" in {t_family, s_family}:
        return "vit"
    return "cnn"


@torch.no_grad()
def eval_accuracy(model: torch.nn.Module, dataloader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        logits = model(data)
        pred = logits.argmax(dim=1)
        total += target.size(0)
        correct += (pred == target).sum().item()
    return correct / max(total, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    args.model_family = resolve_model_family(args)
    if args.model_family != "vit":
        raise ValueError("Boomerang interpolation eval currently supports ViT only.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, eval_loader, num_classes = get_dataloaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        root=args.data_root,
        model_family=args.model_family,
    )

    teacher = TeacherModel(
        model_name=args.teacher_model,
        num_classes=num_classes,
        weights_path=args.teacher_weights,
        pretrained=(args.teacher_weights is None),
        model_family=args.model_family,
    ).to(device)

    student = StudentModel(
        model_name=args.student_model,
        num_classes=num_classes,
        weights_path=args.student_weights,
        model_family=args.model_family,
        num_layers=args.student_num_layers,
    ).to(device)

    kd_model = BoomerangKD(
        teacher=teacher,
        student=student,
        keep_every=args.boomerang_keep_every,
        keep_last_layer=args.boomerang_keep_last_layer,
        student_num_layers=args.student_num_layers,
    ).to(device)

    try:
        checkpoint = torch.load(args.boomerang_checkpoint, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.boomerang_checkpoint, map_location=device)
    kd_model.load_state_dict(checkpoint, strict=True)

    layer_map = kd_model.get_layer_map()
    M = layer_map.student_num_layers

    rows: List[dict] = []

    student_acc = eval_accuracy(kd_model.student, eval_loader, device)
    rows.append(
        {
            "model_type": "student",
            "patched_layers": 0,
            "val_acc": student_acc,
            "inference_params": count_inference_params(kd_model.student.model),
        }
    )

    for K in range(1, M + 1):
        intermediate = kd_model.build_intermediate(
            num_layers_to_patch=K,
            patch_order=args.boomerang_patch_order,
        ).to(device)
        inter_acc = eval_accuracy(intermediate, eval_loader, device)
        rows.append(
            {
                "model_type": "intermediate",
                "patched_layers": K,
                "val_acc": inter_acc,
                "inference_params": count_inference_params(intermediate),
            }
        )

    teacher_acc = eval_accuracy(kd_model.teacher, eval_loader, device)
    rows.append(
        {
            "model_type": "teacher",
            "patched_layers": M,
            "val_acc": teacher_acc,
            "inference_params": count_inference_params(kd_model.teacher.model),
        }
    )

    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, f"{args.experiment_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_type", "patched_layers", "val_acc", "inference_params"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Boomerang interpolation sweep complete.")
    print(f"  Student depth (M): {M}")
    print(f"  Patch order      : {args.boomerang_patch_order}")
    print(f"  CSV              : {csv_path}")
    for row in rows:
        print(
            f"  {row['model_type']:12s} K={int(row['patched_layers']):2d} "
            f"acc={row['val_acc']*100:.2f}% params={int(row['inference_params']):,}"
        )


if __name__ == "__main__":
    main()
