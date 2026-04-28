from __future__ import annotations

import copy
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class BoomerangLayerMap:
    teacher_num_layers: int
    student_num_layers: int
    keep_indices: List[int]
    block_ranges: List[Tuple[int, int]]
    block_end_indices: List[int]


def is_vit_backbone(model: nn.Module) -> bool:
    return (
        hasattr(model, "encoder")
        and hasattr(model.encoder, "layers")
        and hasattr(model, "heads")
        and hasattr(model, "class_token")
        and hasattr(model, "conv_proj")
    )


def get_vit_encoder_layers(model: nn.Module) -> List[nn.Module]:
    if not is_vit_backbone(model):
        raise ValueError("Expected a torchvision VisionTransformer-like model.")
    layers = list(model.encoder.layers.children())
    if not layers:
        raise ValueError("VisionTransformer has no encoder layers.")
    return layers


def set_vit_encoder_layers(model: nn.Module, layers: Sequence[nn.Module]) -> None:
    if not is_vit_backbone(model):
        raise ValueError("Expected a torchvision VisionTransformer-like model.")
    seq = nn.Sequential(OrderedDict((f"encoder_layer_{i}", layer) for i, layer in enumerate(layers)))
    model.encoder.layers = seq


def get_vit_num_layers(model: nn.Module) -> int:
    return len(get_vit_encoder_layers(model))


def infer_model_family(model_name: str) -> str:
    name = model_name.lower()
    if name.startswith("vit"):
        return "vit"
    if name.startswith("resnet") or name.startswith("vgg"):
        return "cnn"
    return "unknown"


def build_keep_indices(
    teacher_num_layers: int,
    student_num_layers: int,
    *,
    keep_every: int = 2,
    keep_last_layer: bool = True,
) -> List[int]:
    if teacher_num_layers <= 0:
        raise ValueError("teacher_num_layers must be > 0")
    if student_num_layers <= 0:
        raise ValueError("student_num_layers must be > 0")
    if student_num_layers > teacher_num_layers:
        raise ValueError("student_num_layers cannot exceed teacher_num_layers")
    if keep_every <= 0:
        raise ValueError("keep_every must be > 0")

    keep = [idx for idx in range(0, teacher_num_layers, keep_every)]
    if not keep:
        keep = [0]

    if keep_last_layer:
        keep[-1] = teacher_num_layers - 1

    keep = sorted(set(keep))

    if len(keep) > student_num_layers:
        keep = keep[: student_num_layers - 1] + [keep[-1]]

    if len(keep) < student_num_layers:
        missing = [i for i in range(teacher_num_layers) if i not in keep]
        for idx in missing:
            keep.append(idx)
            if len(keep) == student_num_layers:
                break

    keep = sorted(keep)

    if keep[0] != 0:
        keep[0] = 0
        keep = sorted(set(keep))

    if len(keep) != student_num_layers:
        raise ValueError(
            f"Could not build exactly {student_num_layers} keep indices from teacher depth {teacher_num_layers}."
        )

    return keep


def build_layer_map(
    teacher_num_layers: int,
    student_num_layers: int,
    *,
    keep_every: int = 2,
    keep_last_layer: bool = True,
) -> BoomerangLayerMap:
    keep = build_keep_indices(
        teacher_num_layers,
        student_num_layers,
        keep_every=keep_every,
        keep_last_layer=keep_last_layer,
    )

    block_ranges: List[Tuple[int, int]] = []
    for i, start_idx in enumerate(keep):
        if i + 1 < len(keep):
            end_idx = keep[i + 1] - 1
        else:
            end_idx = teacher_num_layers - 1
        if end_idx < start_idx:
            raise ValueError(f"Invalid block range ({start_idx}, {end_idx}).")
        block_ranges.append((start_idx, end_idx))

    block_end_indices = [end for _, end in block_ranges]
    return BoomerangLayerMap(
        teacher_num_layers=teacher_num_layers,
        student_num_layers=student_num_layers,
        keep_indices=keep,
        block_ranges=block_ranges,
        block_end_indices=block_end_indices,
    )


def initialize_vit_student_from_teacher(
    teacher_model: nn.Module,
    student_model: nn.Module,
    layer_map: BoomerangLayerMap,
) -> None:
    if not is_vit_backbone(teacher_model) or not is_vit_backbone(student_model):
        raise ValueError("Boomerang initialization currently supports ViT backbones only.")

    teacher_layers = get_vit_encoder_layers(teacher_model)
    student_layers = get_vit_encoder_layers(student_model)

    if len(student_layers) != layer_map.student_num_layers:
        raise ValueError(
            f"Student layer count mismatch: expected {layer_map.student_num_layers}, got {len(student_layers)}"
        )
    if len(teacher_layers) != layer_map.teacher_num_layers:
        raise ValueError(
            f"Teacher layer count mismatch: expected {layer_map.teacher_num_layers}, got {len(teacher_layers)}"
        )

    with torch.no_grad():
        student_model.conv_proj.load_state_dict(teacher_model.conv_proj.state_dict(), strict=True)
        student_model.class_token.copy_(teacher_model.class_token)
        student_model.encoder.pos_embedding.copy_(teacher_model.encoder.pos_embedding)
        student_model.encoder.ln.load_state_dict(teacher_model.encoder.ln.state_dict(), strict=True)
        student_model.heads.load_state_dict(teacher_model.heads.state_dict(), strict=True)

        for student_idx, teacher_idx in enumerate(layer_map.keep_indices):
            student_layers[student_idx].load_state_dict(teacher_layers[teacher_idx].state_dict(), strict=True)


def build_vit_intermediate_from_teacher_student(
    teacher_model: nn.Module,
    student_model: nn.Module,
    layer_map: BoomerangLayerMap,
    *,
    num_layers_to_patch: int,
    patch_order: str = "reverse",
) -> nn.Module:
    if patch_order not in {"reverse", "forward"}:
        raise ValueError("patch_order must be one of {'reverse', 'forward'}")

    if num_layers_to_patch < 0:
        raise ValueError("num_layers_to_patch must be >= 0")
    if num_layers_to_patch > layer_map.student_num_layers:
        raise ValueError(
            f"num_layers_to_patch={num_layers_to_patch} exceeds student depth={layer_map.student_num_layers}."
        )

    teacher_layers = get_vit_encoder_layers(teacher_model)
    student_layers = get_vit_encoder_layers(student_model)

    if patch_order == "reverse":
        patch_indices = list(range(layer_map.student_num_layers - num_layers_to_patch, layer_map.student_num_layers))
    else:
        patch_indices = list(range(0, num_layers_to_patch))

    layers_out: List[nn.Module] = []
    for student_idx in range(layer_map.student_num_layers):
        if student_idx in patch_indices:
            start_idx, end_idx = layer_map.block_ranges[student_idx]
            for teacher_idx in range(start_idx, end_idx + 1):
                layers_out.append(copy.deepcopy(teacher_layers[teacher_idx]))
        else:
            layers_out.append(copy.deepcopy(student_layers[student_idx]))

    intermediate = copy.deepcopy(student_model)
    set_vit_encoder_layers(intermediate, layers_out)

    # For ViT-B on Imagenette we keep identical stems/head dimensions,
    # so we can keep student embedding/head consistently for stable evaluation.
    return intermediate


def count_inference_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
