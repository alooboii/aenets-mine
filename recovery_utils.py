from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from autoencoders import SparseAutoencoder


RESNET_STAGE_ORDER: List[str] = ["layer1", "layer2", "layer3", "layer4"]
RESNET_STEM_NAMES: List[str] = ["conv1", "bn1", "relu", "maxpool"]


@dataclass
class RecoverySiteCheckResult:
    teacher_stage: str
    student_boundary: str
    student_layers: List[str]
    teacher_channels: int
    student_channels: int
    source_param_count: int
    target_param_count: int
    shape_teacher_prefix: Optional[Tuple[int, ...]]
    shape_teacher_trunk_encoded: Optional[Tuple[int, ...]]
    shape_discarded_student_prefix: Optional[Tuple[int, ...]]
    param_budget_ok: bool
    shape_ok: bool
    recoverable: bool
    reason: str
    error: Optional[str] = None


def is_resnet_backbone(model: nn.Module) -> bool:
    return (
        all(hasattr(model, name) for name in RESNET_STEM_NAMES)
        and all(hasattr(model, name) for name in RESNET_STAGE_ORDER)
        and hasattr(model, "fc")
    )


def parse_stage_name(name: str, *, strict_stage_level: bool = True) -> str:
    raw = name.strip()
    base = raw.split("[", 1)[0]
    if base not in RESNET_STAGE_ORDER:
        raise ValueError(
            f"'{name}' is not a valid ResNet stage. Expected one of {RESNET_STAGE_ORDER}."
        )
    if strict_stage_level and raw != base:
        raise ValueError(
            f"Stage-level rules require plain stage names (got '{name}'). "
            f"Use one of {RESNET_STAGE_ORDER}."
        )
    return base


def resolve_student_boundary(
    student_layer_names: Sequence[str], *, strict_stage_level: bool = True
) -> str:
    if not student_layer_names:
        raise ValueError("student_layer_names is empty.")
    first = student_layer_names[0]
    return parse_stage_name(first, strict_stage_level=strict_stage_level)


def infer_student_layers_from_boundary(boundary_stage: str) -> List[str]:
    boundary_idx = RESNET_STAGE_ORDER.index(boundary_stage)
    return RESNET_STAGE_ORDER[boundary_idx:] + ["fc"]


def _build_resnet_stem(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    layers: List[Tuple[str, nn.Module]] = []
    for name in RESNET_STEM_NAMES:
        if not hasattr(model, name):
            raise ValueError(f"Model is missing expected ResNet stem module '{name}'.")
        layers.append((name, getattr(model, name)))
    return layers


def build_teacher_prefix(model: nn.Module, teacher_stage: str) -> nn.Sequential:
    if not is_resnet_backbone(model):
        raise ValueError("ResNet-only utility: teacher model is not ResNet-like.")
    stage = parse_stage_name(teacher_stage, strict_stage_level=True)
    layers = _build_resnet_stem(model)
    stop_idx = RESNET_STAGE_ORDER.index(stage)
    for layer_name in RESNET_STAGE_ORDER[: stop_idx + 1]:
        layers.append((layer_name, getattr(model, layer_name)))
    return nn.Sequential(OrderedDict(layers))


def build_discarded_student_prefix(model: nn.Module, boundary_stage: str) -> nn.Sequential:
    if not is_resnet_backbone(model):
        raise ValueError("ResNet-only utility: student model is not ResNet-like.")
    boundary = parse_stage_name(boundary_stage, strict_stage_level=True)
    layers = _build_resnet_stem(model)
    boundary_idx = RESNET_STAGE_ORDER.index(boundary)
    for layer_name in RESNET_STAGE_ORDER[:boundary_idx]:
        layers.append((layer_name, getattr(model, layer_name)))
    return nn.Sequential(OrderedDict(layers))


def build_retained_student_tail(model: nn.Module, boundary_stage: str) -> nn.Sequential:
    if not is_resnet_backbone(model):
        raise ValueError("ResNet-only utility: student model is not ResNet-like.")
    boundary = parse_stage_name(boundary_stage, strict_stage_level=True)
    boundary_idx = RESNET_STAGE_ORDER.index(boundary)
    layers: List[Tuple[str, nn.Module]] = []
    for layer_name in RESNET_STAGE_ORDER[boundary_idx:]:
        layers.append((layer_name, getattr(model, layer_name)))
    if hasattr(model, "avgpool"):
        layers.append(("avgpool", model.avgpool))
    layers.append(("flatten", nn.Flatten(1)))
    layers.append(("fc", model.fc))
    return nn.Sequential(OrderedDict(layers))


def build_sae_encoder(teacher_channels: int, student_channels: int) -> nn.Sequential:
    sae = SparseAutoencoder(
        teacher_channels=teacher_channels,
        student_channels=student_channels,
        lambda_sparsity=0.0,
    )
    return sae.encoder


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def probe_shapes(
    teacher_prefix: nn.Module,
    sae_encoder: nn.Module,
    discarded_student_prefix: nn.Module,
    device: torch.device,
    *,
    image_size: int = 224,
    batch_size: int = 2,
) -> Dict[str, Tuple[int, ...]]:
    teacher_prefix = teacher_prefix.to(device)
    sae_encoder = sae_encoder.to(device)
    discarded_student_prefix = discarded_student_prefix.to(device)

    teacher_prefix.eval()
    sae_encoder.eval()
    discarded_student_prefix.eval()

    with torch.no_grad():
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        teacher_out = teacher_prefix(x)
        encoded_out = sae_encoder(teacher_out)
        student_prefix_out = discarded_student_prefix(x)

    return {
        "shape_teacher_prefix": tuple(teacher_out.shape),
        "shape_teacher_trunk_encoded": tuple(encoded_out.shape),
        "shape_discarded_student_prefix": tuple(student_prefix_out.shape),
    }


def validate_recovery_site(
    teacher_model: nn.Module,
    student_model: nn.Module,
    *,
    teacher_layer_name: str,
    student_layer_names: Sequence[str],
    teacher_channels: int,
    student_channels: int,
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 2,
) -> RecoverySiteCheckResult:
    teacher_stage = parse_stage_name(teacher_layer_name, strict_stage_level=True)
    student_boundary = resolve_student_boundary(
        student_layer_names, strict_stage_level=True
    )
    student_layers = list(student_layer_names)

    teacher_prefix = build_teacher_prefix(teacher_model, teacher_stage)
    discarded_prefix = build_discarded_student_prefix(student_model, student_boundary)
    sae_encoder = build_sae_encoder(teacher_channels, student_channels)

    source_param_count = count_parameters(teacher_prefix) + count_parameters(sae_encoder)
    target_param_count = count_parameters(discarded_prefix)
    param_budget_ok = source_param_count >= target_param_count

    shape_teacher_prefix: Optional[Tuple[int, ...]] = None
    shape_teacher_trunk_encoded: Optional[Tuple[int, ...]] = None
    shape_discarded_student_prefix: Optional[Tuple[int, ...]] = None
    shape_ok = False
    error: Optional[str] = None

    try:
        shapes = probe_shapes(
            teacher_prefix=teacher_prefix,
            sae_encoder=sae_encoder,
            discarded_student_prefix=discarded_prefix,
            device=device,
            image_size=image_size,
            batch_size=batch_size,
        )
        shape_teacher_prefix = shapes["shape_teacher_prefix"]
        shape_teacher_trunk_encoded = shapes["shape_teacher_trunk_encoded"]
        shape_discarded_student_prefix = shapes["shape_discarded_student_prefix"]
        shape_ok = shape_teacher_trunk_encoded == shape_discarded_student_prefix
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        shape_ok = False

    recoverable = param_budget_ok and shape_ok
    if recoverable:
        reason = "ok"
    elif not shape_ok:
        reason = "shape_fail"
    else:
        reason = "param_budget_fail"

    return RecoverySiteCheckResult(
        teacher_stage=teacher_stage,
        student_boundary=student_boundary,
        student_layers=student_layers,
        teacher_channels=teacher_channels,
        student_channels=student_channels,
        source_param_count=source_param_count,
        target_param_count=target_param_count,
        shape_teacher_prefix=shape_teacher_prefix,
        shape_teacher_trunk_encoded=shape_teacher_trunk_encoded,
        shape_discarded_student_prefix=shape_discarded_student_prefix,
        param_budget_ok=param_budget_ok,
        shape_ok=shape_ok,
        recoverable=recoverable,
        reason=reason,
        error=error,
    )


def copy_prefix_weights(
    target_student_model: nn.Module,
    recovered_prefix: nn.Module,
    *,
    boundary_stage: str,
) -> None:
    target_prefix = build_discarded_student_prefix(target_student_model, boundary_stage)
    target_prefix.load_state_dict(recovered_prefix.state_dict(), strict=True)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def flatten_named_parameters(module: nn.Module, prefix: str) -> List[Tuple[str, torch.Tensor]]:
    flat: List[Tuple[str, torch.Tensor]] = []
    for name, param in module.named_parameters():
        flat.append((f"{prefix}.{name}", param.detach().reshape(-1).cpu()))
    return flat


def topk_prune_and_pack(
    source_modules: Sequence[Tuple[str, nn.Module]],
    target_module: nn.Module,
    *,
    device: torch.device,
) -> Dict[str, int]:
    source_chunks: List[torch.Tensor] = []
    source_counts_by_module: Dict[str, int] = {}
    for module_name, module in source_modules:
        count = 0
        for _, vec in flatten_named_parameters(module, module_name):
            source_chunks.append(vec)
            count += vec.numel()
        source_counts_by_module[module_name] = count

    if not source_chunks:
        raise ValueError("No source parameters found for prune-and-pack.")

    source_flat = torch.cat(source_chunks, dim=0)
    target_params = list(target_module.parameters())
    target_count = sum(param.numel() for param in target_params)
    source_count = source_flat.numel()

    if source_count < target_count:
        raise ValueError(
            f"Source parameter pool is too small: source={source_count}, target={target_count}."
        )

    abs_vals = source_flat.abs()
    sorted_idx = torch.argsort(abs_vals, descending=True)
    top_idx = sorted_idx[:target_count]
    top_idx = torch.sort(top_idx).values
    kept_values = source_flat[top_idx]

    offset = 0
    with torch.no_grad():
        for param in target_params:
            n = param.numel()
            chunk = kept_values[offset : offset + n].to(device=device, dtype=param.dtype)
            param.copy_(chunk.view_as(param))
            offset += n

    prune_ratio = 1.0 - (target_count / source_count)
    return {
        "source_count": int(source_count),
        "target_count": int(target_count),
        "kept_count": int(target_count),
        "pruned_count": int(source_count - target_count),
        "prune_ratio_ppm": int(prune_ratio * 1_000_000),
        **{f"source_{k}_count": int(v) for k, v in source_counts_by_module.items()},
    }
