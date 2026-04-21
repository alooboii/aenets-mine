from __future__ import annotations

import heapq
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


@dataclass(frozen=True)
class _ParamMetadata:
    family: str
    role: str
    stage: str
    kernel_tag: str


@dataclass
class _TargetParamRecord:
    index: int
    name: str
    param: torch.nn.Parameter
    metadata: _ParamMetadata


@dataclass
class _SourceBucket:
    values: torch.Tensor
    scores: torch.Tensor
    cursor: int = 0

    def has_remaining(self) -> bool:
        return self.cursor < self.values.numel()

    def remaining(self) -> int:
        return self.values.numel() - self.cursor


_CONV_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)
_NORM_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
)
_STAGE_TO_DEPTH = {
    "other": -1,
    "stem": 0,
    "layer1": 1,
    "layer2": 2,
    "layer3": 3,
    "layer4": 4,
    "adapter": 5,
}


def _module_family(module: nn.Module) -> str:
    if isinstance(module, _CONV_TYPES):
        return "conv"
    if isinstance(module, _NORM_TYPES):
        return "norm"
    if isinstance(module, nn.Linear):
        return "linear"
    return "other"


def _param_role(param_name: str) -> str:
    leaf = param_name.rsplit(".", 1)[-1]
    if leaf in {"weight", "bias"}:
        return leaf
    return "other"


def _stage_from_param_name(local_param_name: str, module_name: str) -> str:
    if module_name == "sae_encoder":
        return "adapter"

    segments = local_param_name.split(".")
    for seg in segments:
        if seg in RESNET_STAGE_ORDER:
            return seg

    if segments and segments[0] in RESNET_STEM_NAMES:
        return "stem"
    return "other"


def _kernel_tag(param: torch.Tensor, family: str, role: str) -> str:
    if family != "conv" or role != "weight":
        return "na"
    if param.ndim < 2:
        return "other"
    if param.ndim >= 4:
        k_h = int(param.shape[-2])
        k_w = int(param.shape[-1])
    else:
        k_h = int(param.shape[-1])
        k_w = int(param.shape[-1])
    if k_h == k_w and k_h in {1, 3, 7}:
        return f"k{k_h}"
    return "other"


def _stable_desc_argsort(values: torch.Tensor) -> torch.Tensor:
    try:
        return torch.argsort(values, descending=True, stable=True)
    except TypeError:
        return torch.argsort(values, descending=True)


def _collect_target_param_records(target_module: nn.Module) -> List[_TargetParamRecord]:
    named_modules = dict(target_module.named_modules())
    out: List[_TargetParamRecord] = []
    for idx, (name, param) in enumerate(target_module.named_parameters()):
        module_path = name.rsplit(".", 1)[0] if "." in name else ""
        owner = named_modules.get(module_path, target_module)
        family = _module_family(owner)
        role = _param_role(name)
        stage = _stage_from_param_name(name, module_name="target")
        kernel_tag = _kernel_tag(param.detach(), family=family, role=role)
        out.append(
            _TargetParamRecord(
                index=idx,
                name=name,
                param=param,
                metadata=_ParamMetadata(
                    family=family,
                    role=role,
                    stage=stage,
                    kernel_tag=kernel_tag,
                ),
            )
        )
    return out


def _collect_source_buckets(
    source_modules: Sequence[Tuple[str, nn.Module]],
) -> Tuple[Dict[Tuple[str, str, str, str], _SourceBucket], Dict[str, int], Dict[str, int]]:
    raw_values: Dict[Tuple[str, str, str, str], List[torch.Tensor]] = {}
    raw_scores: Dict[Tuple[str, str, str, str], List[torch.Tensor]] = {}
    source_counts_by_module: Dict[str, int] = {}
    source_counts_by_family_role: Dict[str, int] = {}

    for module_name, module in source_modules:
        named_modules = dict(module.named_modules())
        count = 0
        for param_name, param in module.named_parameters():
            module_path = param_name.rsplit(".", 1)[0] if "." in param_name else ""
            owner = named_modules.get(module_path, module)
            family = _module_family(owner)
            role = _param_role(param_name)
            stage = _stage_from_param_name(param_name, module_name=module_name)
            kernel_tag = _kernel_tag(param.detach(), family=family, role=role)
            key = (family, role, stage, kernel_tag)

            vec = param.detach().reshape(-1).cpu()
            if vec.numel() == 0:
                continue

            abs_vals = vec.abs()
            denom = abs_vals.mean() + 1e-12
            norm_abs = abs_vals / denom

            raw_values.setdefault(key, []).append(vec)
            raw_scores.setdefault(key, []).append(norm_abs)

            count += vec.numel()
            fr_key = f"{family}:{role}"
            source_counts_by_family_role[fr_key] = (
                source_counts_by_family_role.get(fr_key, 0) + int(vec.numel())
            )

        source_counts_by_module[module_name] = count

    if not raw_values:
        raise ValueError("No source parameters found for family-depth prune-and-pack.")

    buckets: Dict[Tuple[str, str, str, str], _SourceBucket] = {}
    for key in sorted(raw_values.keys()):
        values = torch.cat(raw_values[key], dim=0)
        scores = torch.cat(raw_scores[key], dim=0)
        order = _stable_desc_argsort(scores)
        buckets[key] = _SourceBucket(values=values[order], scores=scores[order], cursor=0)

    return buckets, source_counts_by_module, source_counts_by_family_role


def _target_counts_by_family_role(
    target_records: Sequence[_TargetParamRecord],
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in target_records:
        fr_key = f"{rec.metadata.family}:{rec.metadata.role}"
        counts[fr_key] = counts.get(fr_key, 0) + int(rec.param.numel())
    return counts


def _select_top_from_buckets(
    bucket_keys: Sequence[Tuple[str, str, str, str]],
    need: int,
    buckets: Dict[Tuple[str, str, str, str], _SourceBucket],
) -> Tuple[torch.Tensor, int, Dict[str, int], int, float]:
    if need <= 0:
        return torch.empty(0), 0, {}, 0, 0.0

    active_keys = [k for k in bucket_keys if k in buckets and buckets[k].has_remaining()]
    if not active_keys:
        return torch.empty(0), 0, {}, 0, 0.0

    # Deterministic tie-break on key order.
    key_order = {k: i for i, k in enumerate(sorted(active_keys))}

    heap: List[Tuple[float, int, Tuple[str, str, str, str]]] = []
    for key in active_keys:
        bucket = buckets[key]
        score = float(bucket.scores[bucket.cursor].item())
        heapq.heappush(heap, (-score, key_order[key], key))

    selected_values: List[torch.Tensor] = []
    selected = 0
    per_key_counts: Dict[str, int] = {}
    score_sum = 0.0

    while heap and selected < need:
        neg_score, _, key = heapq.heappop(heap)
        bucket = buckets[key]
        idx = bucket.cursor
        selected_values.append(bucket.values[idx : idx + 1])
        selected += 1
        score_sum += -neg_score
        key_label = "|".join(key)
        per_key_counts[key_label] = per_key_counts.get(key_label, 0) + 1
        bucket.cursor += 1
        if bucket.has_remaining():
            next_score = float(bucket.scores[bucket.cursor].item())
            heapq.heappush(heap, (-next_score, key_order[key], key))

    if selected_values:
        values = torch.cat(selected_values, dim=0)
    else:
        values = torch.empty(0)

    abs_sum = float(values.abs().sum().item()) if values.numel() > 0 else 0.0
    return values, selected, per_key_counts, int(values.numel()), abs_sum + score_sum


def family_depth_prune_and_pack(
    source_modules: Sequence[Tuple[str, nn.Module]],
    target_module: nn.Module,
    *,
    device: torch.device,
) -> Dict[str, object]:
    buckets, source_counts_by_module, source_counts_by_family_role = _collect_source_buckets(
        source_modules
    )
    target_records = _collect_target_param_records(target_module)
    target_counts_by_family_role = _target_counts_by_family_role(target_records)

    deficits: List[str] = []
    for fr_key in sorted(target_counts_by_family_role):
        src = source_counts_by_family_role.get(fr_key, 0)
        tgt = target_counts_by_family_role[fr_key]
        if src < tgt:
            deficits.append(f"{fr_key} (source={src}, target={tgt}, deficit={tgt-src})")
    if deficits:
        raise ValueError(
            "family-depth prune-and-pack source budget is insufficient for "
            "target family/role groups: " + "; ".join(deficits)
        )

    fr_to_bucket_keys: Dict[Tuple[str, str], List[Tuple[str, str, str, str]]] = {}
    for key in sorted(buckets.keys()):
        fr_to_bucket_keys.setdefault((key[0], key[1]), []).append(key)

    def stage_depth(stage_name: str) -> int:
        return _STAGE_TO_DEPTH.get(stage_name, _STAGE_TO_DEPTH["other"])

    # Mild back-loaded preference: fill deeper target params first.
    target_records = sorted(
        target_records,
        key=lambda rec: (-stage_depth(rec.metadata.stage), rec.index),
    )

    priority_selected_counts = {"level1": 0, "level2": 0, "level3": 0, "level4": 0}
    per_priority_key_counts = {
        "level1": {},
        "level2": {},
        "level3": {},
        "level4": {},
    }
    selection_abs_plus_score_sum = 0.0
    selection_elem_count = 0

    with torch.no_grad():
        for rec in target_records:
            meta = rec.metadata
            fr_keys = fr_to_bucket_keys.get((meta.family, meta.role), [])
            if not fr_keys:
                raise ValueError(
                    f"No source bucket exists for target tensor '{rec.name}' "
                    f"family={meta.family}, role={meta.role}."
                )

            same_stage_keys = [k for k in fr_keys if k[2] == meta.stage]
            same_stage_same_kernel = [k for k in same_stage_keys if k[3] == meta.kernel_tag]
            same_stage_any_kernel = [k for k in same_stage_keys]

            other_stage_keys = [k for k in fr_keys if k[2] != meta.stage]
            other_stage_keys_sorted = sorted(
                other_stage_keys,
                key=lambda k: (
                    abs(stage_depth(k[2]) - stage_depth(meta.stage)),
                    -stage_depth(k[2]),
                    0 if k[3] == meta.kernel_tag else 1,
                    k[3],
                ),
            )

            level4_any = sorted(
                fr_keys,
                key=lambda k: (
                    abs(stage_depth(k[2]) - stage_depth(meta.stage)),
                    -stage_depth(k[2]),
                    0 if k[3] == meta.kernel_tag else 1,
                    k[3],
                ),
            )

            needed = int(rec.param.numel())
            chunks: List[torch.Tensor] = []

            levels = [
                ("level1", same_stage_same_kernel),
                ("level2", same_stage_any_kernel),
                ("level3", other_stage_keys_sorted),
                ("level4", level4_any),
            ]

            for level_name, level_keys in levels:
                if needed <= 0:
                    break
                selected_values, selected_n, key_counts, elem_n, checksum_part = _select_top_from_buckets(
                    level_keys, needed, buckets
                )
                if selected_n > 0:
                    chunks.append(selected_values)
                    needed -= selected_n
                    priority_selected_counts[level_name] += selected_n
                    selection_elem_count += elem_n
                    selection_abs_plus_score_sum += checksum_part
                    level_counter = per_priority_key_counts[level_name]
                    for key_label, val in key_counts.items():
                        level_counter[key_label] = level_counter.get(key_label, 0) + int(val)

            if needed != 0:
                raise ValueError(
                    f"family-depth prune-and-pack could not fill target tensor '{rec.name}' "
                    f"(family={meta.family}, role={meta.role}, stage={meta.stage}). "
                    f"remaining={needed}."
                )

            selected_flat = torch.cat(chunks, dim=0)
            if selected_flat.numel() != rec.param.numel():
                raise RuntimeError(
                    f"Internal error while packing '{rec.name}': "
                    f"selected={selected_flat.numel()} target={rec.param.numel()}."
                )
            rec.param.copy_(
                selected_flat.to(device=device, dtype=rec.param.dtype).view_as(rec.param)
            )

    source_count = sum(source_counts_by_family_role.values())
    target_count = sum(target_counts_by_family_role.values())
    if source_count < target_count:
        raise ValueError(
            f"Source parameter pool is too small: source={source_count}, target={target_count}."
        )
    prune_ratio = 1.0 - (target_count / max(source_count, 1))

    family_role_margin = {
        fr: int(source_counts_by_family_role.get(fr, 0) - target_counts_by_family_role.get(fr, 0))
        for fr in sorted(target_counts_by_family_role.keys())
    }

    return {
        "init_method": "family_depth_stage",
        "source_count": int(source_count),
        "target_count": int(target_count),
        "kept_count": int(target_count),
        "pruned_count": int(source_count - target_count),
        "prune_ratio_ppm": int(prune_ratio * 1_000_000),
        "same_type_spillover_selected_count": int(priority_selected_counts["level4"]),
        "priority_selected_counts": {k: int(v) for k, v in priority_selected_counts.items()},
        "priority_bucket_usage_counts": per_priority_key_counts,
        "family_role_source_counts": {
            k: int(v) for k, v in sorted(source_counts_by_family_role.items())
        },
        "family_role_target_counts": {
            k: int(v) for k, v in sorted(target_counts_by_family_role.items())
        },
        "family_role_margin_counts": family_role_margin,
        "selection_checksum_like": {
            "selected_elem_count": int(selection_elem_count),
            "abs_plus_score_sum_scaled": int(selection_abs_plus_score_sum * 1_000_000),
        },
        **{f"source_{k}_count": int(v) for k, v in source_counts_by_module.items()},
    }


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
