from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from boomerang_utils import (
    BoomerangLayerMap,
    build_layer_map,
    build_vit_intermediate_from_teacher_student,
    get_vit_encoder_layers,
    initialize_vit_student_from_teacher,
    is_vit_backbone,
)


class _FeatureHooks:
    def __init__(self, named_layers):
        self.features = OrderedDict()
        self.hooks = []

        def hook_fn(name):
            def _hook(_module, _input, output):
                self.features[name] = output

            return _hook

        for name, layer in named_layers:
            self.hooks.append(layer.register_forward_hook(hook_fn(name)))

    def clear(self):
        self.features.clear()

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class BoomerangKD(nn.Module):
    """
    Boomerang distillation on ViT backbones.

    Objective pieces:
      - CE is handled by the training loop
      - KL is handled by the training loop (with temperature scaling)
      - This module returns the summed layer cosine alignment loss as aux
    """

    def __init__(
        self,
        teacher,
        student,
        *,
        keep_every: int = 2,
        keep_last_layer: bool = True,
        student_num_layers: Optional[int] = None,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student

        if not is_vit_backbone(self.teacher.model) or not is_vit_backbone(self.student.model):
            raise ValueError(
                "BoomerangKD currently supports ViT backbones only. "
                "Use a torchvision ViT teacher/student pair."
            )

        teacher_depth = len(get_vit_encoder_layers(self.teacher.model))
        student_depth = len(get_vit_encoder_layers(self.student.model))
        if student_num_layers is not None and student_num_layers != student_depth:
            raise ValueError(
                f"student_num_layers={student_num_layers} does not match built student depth={student_depth}."
            )

        self.layer_map = build_layer_map(
            teacher_num_layers=teacher_depth,
            student_num_layers=student_depth,
            keep_every=keep_every,
            keep_last_layer=keep_last_layer,
        )

        # Deterministic boomerang init from teacher layers.
        initialize_vit_student_from_teacher(self.teacher.model, self.student.model, self.layer_map)

        # Freeze teacher.
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        teacher_layers = get_vit_encoder_layers(self.teacher.model)
        student_layers = get_vit_encoder_layers(self.student.model)

        teacher_named = []
        student_named = []
        for i, block_end_idx in enumerate(self.layer_map.block_end_indices):
            teacher_named.append((f"teacher_block_end_{i}", teacher_layers[block_end_idx]))
            student_named.append((f"student_layer_{i}", student_layers[i]))

        self.teacher_hooks = _FeatureHooks(teacher_named)
        self.student_hooks = _FeatureHooks(student_named)

    def get_layer_map(self) -> BoomerangLayerMap:
        return self.layer_map

    def build_intermediate(self, num_layers_to_patch: int, patch_order: str = "reverse") -> nn.Module:
        return build_vit_intermediate_from_teacher_student(
            teacher_model=self.teacher.model,
            student_model=self.student.model,
            layer_map=self.layer_map,
            num_layers_to_patch=num_layers_to_patch,
            patch_order=patch_order,
        )

    def _cosine_alignment_loss(self) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for i in range(self.layer_map.student_num_layers):
            t = self.teacher_hooks.features.get(f"teacher_block_end_{i}")
            s = self.student_hooks.features.get(f"student_layer_{i}")
            if t is None or s is None:
                raise RuntimeError(
                    f"Missing hook features for mapping index {i}. "
                    "Check ViT layer hook registration."
                )
            if t.shape != s.shape:
                raise RuntimeError(
                    f"Hook shape mismatch at layer {i}: teacher={tuple(t.shape)} student={tuple(s.shape)}"
                )
            # Mean across tokens and batch.
            cos = F.cosine_similarity(s, t, dim=-1)
            losses.append(1.0 - cos.mean())

        return torch.stack(losses).sum()

    def forward(self, x):
        self.teacher_hooks.clear()
        self.student_hooks.clear()

        with torch.no_grad():
            teacher_logits = self.teacher(x)

        student_logits = self.student(x)
        cos_loss = self._cosine_alignment_loss()

        return teacher_logits, student_logits, cos_loss

    def remove_hooks(self) -> None:
        self.teacher_hooks.remove()
        self.student_hooks.remove()
