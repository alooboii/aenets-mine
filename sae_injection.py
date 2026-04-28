from collections import OrderedDict
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from autoencoders import SparseAutoencoder
from boomerang_utils import get_vit_encoder_layers, is_vit_backbone


class FeatureHooks:
    """
    Helper class to extract intermediate features from a network using forward hooks.
    """

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


class _ViTTeacherPrefix(nn.Module):
    """Teacher prefix for torchvision ViT: embed+class token+pos/dropout+first N blocks."""

    def __init__(self, vit_model: nn.Module, num_blocks: int):
        super().__init__()
        self.conv_proj = vit_model.conv_proj
        self.class_token = vit_model.class_token
        self.pos_embedding = vit_model.encoder.pos_embedding
        self.dropout = vit_model.encoder.dropout

        layers = get_vit_encoder_layers(vit_model)
        if num_blocks < 0 or num_blocks > len(layers):
            raise ValueError(
                f"Invalid ViT teacher boundary num_blocks={num_blocks}; "
                f"expected in [0, {len(layers)}]."
            )
        if num_blocks == 0:
            self.blocks = nn.Identity()
        else:
            self.blocks = nn.Sequential(
                OrderedDict((f"encoder_layer_{i}", layers[i]) for i in range(num_blocks))
            )

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "conv_proj"):
            n = x.shape[0]
            x = self.conv_proj(x)
            x = x.reshape(n, x.shape[1], -1)
            x = x.permute(0, 2, 1)
            return x
        raise RuntimeError("ViT conv_proj is missing on teacher prefix.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._process_input(x)
        n = x.shape[0]
        cls = self.class_token.expand(n, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, : x.size(1), :]
        x = self.dropout(x)
        x = self.blocks(x)
        return x


class _ViTStudentTail(nn.Module):
    """Student tail for torchvision ViT: remaining blocks + final norm + classifier head."""

    def __init__(self, vit_model: nn.Module, start_block: int):
        super().__init__()
        layers = get_vit_encoder_layers(vit_model)
        depth = len(layers)
        if start_block < 0 or start_block > depth:
            raise ValueError(
                f"Invalid ViT student boundary start_block={start_block}; expected in [0, {depth}]."
            )

        tail_layers = layers[start_block:]
        if tail_layers:
            self.blocks = nn.Sequential(
                OrderedDict((f"encoder_layer_{i + start_block}", layer) for i, layer in enumerate(tail_layers))
            )
        else:
            self.blocks = nn.Identity()

        self.final_norm = vit_model.encoder.ln
        self.heads = vit_model.heads

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.blocks(tokens)
        x = self.final_norm(x)
        cls = x[:, 0]
        logits = self.heads(cls)
        return logits


class HybridModel(nn.Module):
    """
    Hybrid model combining partial teacher + SAE + partial student tail.
    """

    def __init__(self, teacher_trunk, sae_adapter, student_trunk, freeze_teacher):
        super().__init__()
        self.teacher_trunk = teacher_trunk
        self.sae_adapter = sae_adapter
        self.student_trunk = student_trunk

        if freeze_teacher:
            print("Teacher trunk parameters frozen.")
            for param in self.teacher_trunk.parameters():
                param.requires_grad = False
            self.teacher_trunk.eval()
        else:
            for param in self.teacher_trunk.parameters():
                param.requires_grad = True

        self.freeze_teacher = freeze_teacher

    def forward(self, x):
        if self.freeze_teacher:
            with torch.no_grad():
                teacher_features = self.teacher_trunk(x)
        else:
            teacher_features = self.teacher_trunk(x)

        _recon, latent, sae_loss = self.sae_adapter(teacher_features)
        student_logits = self.student_trunk(latent)
        return student_logits, sae_loss


class SAEInjection(nn.Module):
    """
    SAE-based feature injection for KD.

    CNN mode (existing behavior):
      Teacher prefix up to --teacher-layer -> SAE(conv2d) -> student trunk from --student-layers.

    ViT mode:
      Teacher prefix (embed + first t blocks) -> SAE(token_linear) -> student tail from boundary.
    """

    def __init__(
        self,
        teacher,
        student,
        teacher_layer_name=None,
        teacher_channels=None,
        student_layer_names=None,
        student_channels=None,
        sparsity=1e-4,
        freeze_teacher=True,
        teacher_boundary: Optional[int] = None,
        student_boundary: Optional[int] = None,
        sae_adapter_type: str = "auto",
        model_family: str = "auto",
    ):
        super().__init__()

        self.teacher = teacher
        self.student = student
        self.sparsity = sparsity
        self.teacher_layer_name = teacher_layer_name
        self.student_layer_names = student_layer_names
        self.freeze_teacher = freeze_teacher
        self.teacher_boundary = teacher_boundary
        self.student_boundary = student_boundary
        self.sae_adapter_type = sae_adapter_type
        self.model_family = model_family

        self.backbone_family = self._infer_backbone_family(
            teacher.model,
            student.model,
            model_family=model_family,
        )

        self.hybrid_model, build_info = self._create_hybrid_model(
            teacher=teacher,
            student=student,
            teacher_layer_name=teacher_layer_name,
            teacher_channels=teacher_channels,
            student_layer_names=student_layer_names,
            student_channels=student_channels,
            sparsity=sparsity,
            freeze_teacher=freeze_teacher,
            teacher_boundary=teacher_boundary,
            student_boundary=student_boundary,
            sae_adapter_type=sae_adapter_type,
            backbone_family=self.backbone_family,
        )
        self.build_info = build_info

        print(f"\n{'='*72}")
        print("SAEInjection: Hybrid Architecture Created")
        print(f"{'='*72}")
        print(f"Backbone family          : {self.backbone_family}")
        if self.backbone_family == "vit":
            print(f"Teacher boundary (blocks): {build_info['teacher_boundary']}")
            print(f"Student boundary (blocks): {build_info['student_boundary']}")
            print(f"Teacher hidden dim       : {build_info['teacher_channels']}")
            print(f"Student hidden dim       : {build_info['student_channels']}")
            print(
                "Architecture              : "
                f"Teacher[prefix@{build_info['teacher_boundary']}] -> SAE({build_info['adapter_type']}) -> "
                f"Student[tail@{build_info['student_boundary']}:]"
            )
        else:
            print(f"Teacher hint layer       : {teacher_layer_name} (channels: {build_info['teacher_channels']})")
            print(f"Student layers           : {build_info['student_layers']}")
            print(f"Student SAE input ch     : {build_info['student_channels']}")
            print(
                "Architecture              : "
                f"Teacher[...->{teacher_layer_name}] -> SAE({build_info['adapter_type']}) -> "
                f"Student{build_info['student_layers']}"
            )

        print(f"Teacher frozen           : {freeze_teacher}")
        print(f"Sparsity coefficient     : {sparsity}")
        print(
            f"Total params [hybrid]    : "
            f"{sum(p.numel() for p in self.hybrid_model.parameters()) * 1e-6:.2f}M"
        )
        print(
            f"Total params [teacher]   : "
            f"{sum(p.numel() for p in self.hybrid_model.teacher_trunk.parameters()) * 1e-6:.2f}M"
        )
        print(
            f"Total params [student]   : "
            f"{sum(p.numel() for p in self.hybrid_model.student_trunk.parameters()) * 1e-6:.2f}M"
        )
        print(
            f"Total params [SAE]       : "
            f"{sum(p.numel() for p in self.hybrid_model.sae_adapter.parameters()) * 1e-6:.2f}M"
        )
        print(f"{'='*72}\n")

    def _infer_backbone_family(self, teacher_model: nn.Module, student_model: nn.Module, model_family: str) -> str:
        if model_family != "auto":
            if model_family not in {"cnn", "vit"}:
                raise ValueError("model_family must be one of {'auto', 'cnn', 'vit'}.")
            return model_family

        teacher_is_vit = is_vit_backbone(teacher_model)
        student_is_vit = is_vit_backbone(student_model)
        if teacher_is_vit != student_is_vit:
            raise ValueError(
                "Teacher and student backbone families do not match "
                f"(teacher_vit={teacher_is_vit}, student_vit={student_is_vit})."
            )
        return "vit" if teacher_is_vit else "cnn"

    def _resolve_sae_adapter_type(self, backbone_family: str, requested: str) -> str:
        if requested == "auto":
            return "token_linear" if backbone_family == "vit" else "conv2d"
        if requested not in {"conv2d", "token_linear"}:
            raise ValueError(
                f"Unsupported SAE adapter type '{requested}'. "
                "Expected one of {'auto', 'conv2d', 'token_linear'}."
            )
        return requested

    def _default_teacher_boundary(
        self,
        teacher_depth: int,
        student_depth: int,
        student_boundary: int,
    ) -> int:
        if student_depth <= 0:
            raise ValueError("student_depth must be > 0 for ViT boundary mapping.")
        mapped = int(round((student_boundary / student_depth) * teacher_depth))
        if student_boundary > 0 and mapped == 0:
            mapped = 1
        return max(0, min(mapped, teacher_depth))

    def _create_hybrid_model(
        self,
        teacher,
        student,
        teacher_layer_name,
        teacher_channels,
        student_layer_names,
        student_channels,
        sparsity,
        freeze_teacher,
        teacher_boundary,
        student_boundary,
        sae_adapter_type,
        backbone_family,
    ) -> Tuple[HybridModel, dict]:
        if backbone_family == "vit":
            teacher_depth = len(get_vit_encoder_layers(teacher.model))
            student_depth = len(get_vit_encoder_layers(student.model))

            if student_boundary is None:
                student_boundary = student_depth // 2
            if not (0 <= student_boundary <= student_depth):
                raise ValueError(
                    f"Invalid --aenets-student-boundary={student_boundary}; "
                    f"expected in [0, {student_depth}]."
                )

            if teacher_boundary is None:
                teacher_boundary = self._default_teacher_boundary(
                    teacher_depth=teacher_depth,
                    student_depth=student_depth,
                    student_boundary=student_boundary,
                )
            if not (0 <= teacher_boundary <= teacher_depth):
                raise ValueError(
                    f"Invalid --aenets-teacher-boundary={teacher_boundary}; "
                    f"expected in [0, {teacher_depth}]."
                )

            inferred_teacher_channels = int(getattr(teacher.model, "hidden_dim"))
            inferred_student_channels = int(getattr(student.model, "hidden_dim"))
            teacher_channels_eff = inferred_teacher_channels if teacher_channels is None else int(teacher_channels)
            student_channels_eff = inferred_student_channels if student_channels is None else int(student_channels)

            if teacher_channels is not None and teacher_channels_eff != inferred_teacher_channels:
                print(
                    "Warning: --teacher-channels does not match ViT hidden_dim "
                    f"({teacher_channels_eff} vs {inferred_teacher_channels}). Using provided value."
                )
            if student_channels is not None and student_channels_eff != inferred_student_channels:
                print(
                    "Warning: --student-channels does not match ViT hidden_dim "
                    f"({student_channels_eff} vs {inferred_student_channels}). Using provided value."
                )

            adapter_type = self._resolve_sae_adapter_type(backbone_family, sae_adapter_type)
            teacher_trunk = self._build_teacher_trunk_vit(teacher.model, teacher_boundary)
            student_trunk = self._build_student_trunk_vit(student.model, student_boundary)
            sae_adapter = SparseAutoencoder(
                teacher_channels_eff,
                student_channels_eff,
                sparsity,
                adapter_type=adapter_type,
            )
            # Early shape check for ViT boundary compatibility.
            try:
                with torch.no_grad():
                    probe_device = next(teacher_trunk.parameters()).device
                    probe = torch.randn(1, 3, 224, 224, device=probe_device)
                    teacher_tokens = teacher_trunk(probe)
                    _recon, latent_tokens, _loss = sae_adapter(teacher_tokens)
                    _ = student_trunk(latent_tokens)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    "ViT SAE injection boundary/shape check failed. "
                    f"teacher_boundary={teacher_boundary}, student_boundary={student_boundary}, "
                    f"teacher_channels={teacher_channels_eff}, student_channels={student_channels_eff}. "
                    f"Original error: {exc}"
                ) from exc

            build_info = {
                "teacher_boundary": int(teacher_boundary),
                "student_boundary": int(student_boundary),
                "teacher_channels": int(teacher_channels_eff),
                "student_channels": int(student_channels_eff),
                "adapter_type": adapter_type,
            }
        else:
            if teacher_layer_name is None:
                raise ValueError("teacher_layer_name is required for CNN SAE injection.")
            if not student_layer_names:
                raise ValueError("student_layer_names is required for CNN SAE injection.")
            if teacher_channels is None or student_channels is None:
                raise ValueError(
                    "teacher_channels and student_channels are required for CNN SAE injection."
                )

            adapter_type = self._resolve_sae_adapter_type(backbone_family, sae_adapter_type)
            teacher_trunk = self._build_teacher_trunk_cnn(teacher.model, teacher_layer_name)
            student_trunk = self._build_student_trunk_cnn(student.model, student_layer_names)
            sae_adapter = SparseAutoencoder(
                teacher_channels,
                student_channels,
                sparsity,
                adapter_type=adapter_type,
            )
            build_info = {
                "teacher_channels": int(teacher_channels),
                "student_channels": int(student_channels),
                "student_layers": list(student_layer_names),
                "adapter_type": adapter_type,
            }

        hybrid_model = HybridModel(
            teacher_trunk=teacher_trunk,
            sae_adapter=sae_adapter,
            student_trunk=student_trunk,
            freeze_teacher=freeze_teacher,
        )
        return hybrid_model, build_info

    def _build_teacher_trunk_vit(self, teacher_model: nn.Module, teacher_boundary: int) -> nn.Module:
        if not is_vit_backbone(teacher_model):
            raise ValueError("_build_teacher_trunk_vit expected a ViT backbone.")
        return _ViTTeacherPrefix(teacher_model, num_blocks=teacher_boundary)

    def _build_student_trunk_vit(self, student_model: nn.Module, student_boundary: int) -> nn.Module:
        if not is_vit_backbone(student_model):
            raise ValueError("_build_student_trunk_vit expected a ViT backbone.")
        return _ViTStudentTail(student_model, start_block=student_boundary)

    def _build_teacher_trunk_cnn(self, teacher_model, hint_layer_name):
        """
        Extract teacher layers up to and including the hint layer.
        Supports ResNet and VGG.
        """
        layers = []

        is_vgg = hasattr(teacher_model, "features") and hasattr(teacher_model, "classifier")
        is_resnet = hasattr(teacher_model, "layer1")

        if is_resnet:
            initial_layer_names = ["conv1", "bn1", "relu", "maxpool"]
            for name in initial_layer_names:
                if hasattr(teacher_model, name):
                    layers.append((name, getattr(teacher_model, name)))

            hint_base = hint_layer_name.split("[")[0]
            hint_idx = None
            if "[" in hint_layer_name:
                hint_idx = int(hint_layer_name.split("[")[1].rstrip("]"))

            layer_order = ["layer1", "layer2", "layer3", "layer4"]
            for layer_name in layer_order:
                if hasattr(teacher_model, layer_name):
                    layer_block = getattr(teacher_model, layer_name)

                    if layer_name == hint_base:
                        if hint_idx is not None and isinstance(layer_block, nn.Sequential):
                            for i in range(hint_idx + 1):
                                layers.append((f"{layer_name}[{i}]", layer_block[i]))
                        else:
                            layers.append((layer_name, layer_block))
                        break
                    layers.append((layer_name, layer_block))

                    if layer_order.index(layer_name) >= layer_order.index(hint_base):
                        break

        elif is_vgg:
            if hint_layer_name.startswith("features"):
                features_block = teacher_model.features
                if "[" in hint_layer_name:
                    hint_idx = int(hint_layer_name.split("[")[1].rstrip("]"))
                    for i in range(hint_idx + 1):
                        layers.append((f"features[{i}]", features_block[i]))
                else:
                    layers.append(("features", features_block))

            elif hint_layer_name.startswith("classifier"):
                layers.append(("features", teacher_model.features))
                layers.append(("avgpool", teacher_model.avgpool))
                layers.append(("flatten", nn.Flatten(1)))

                classifier_block = teacher_model.classifier
                if "[" in hint_layer_name:
                    hint_idx = int(hint_layer_name.split("[")[1].rstrip("]"))
                    for i in range(hint_idx + 1):
                        layers.append((f"classifier[{i}]", classifier_block[i]))
                else:
                    layers.append(("classifier", classifier_block))
            else:
                raise ValueError(f"Invalid VGG hint layer name: {hint_layer_name}")

        else:
            raise ValueError("Unsupported architecture (not ResNet, VGG, or ViT)")

        return nn.Sequential(OrderedDict(layers))

    def _build_student_trunk_cnn(self, student_model, layer_names_to_include: Sequence[str]):
        """
        Extract only the specified student layers for CNN backbones.
        Supports ResNet and VGG.
        """
        layers = []

        is_vgg = hasattr(student_model, "features") and hasattr(student_model, "classifier")
        is_resnet = hasattr(student_model, "layer1")

        if is_resnet:
            all_layer_names = [
                "layer1",
                "layer2",
                "layer3",
                "layer4",
                "avgpool",
                "global_pool",
                "flatten",
                "fc",
                "classifier",
            ]
            last_conv_layer = None

            for layer_name in all_layer_names:
                if layer_name in layer_names_to_include:
                    if hasattr(student_model, layer_name):
                        layer = getattr(student_model, layer_name)
                        if layer_name in ["layer1", "layer2", "layer3", "layer4"]:
                            last_conv_layer = layer_name
                        layers.append((layer_name, layer))
                    elif layer_name == "flatten":
                        layers.append(("flatten", nn.Flatten(1)))
                    else:
                        print(f"Warning: Requested layer '{layer_name}' not found in student model")

            if "flatten" in layer_names_to_include:
                has_pooling = any(name in ["avgpool", "global_pool"] for name, _ in layers)
                if not has_pooling:
                    if hasattr(student_model, "avgpool"):
                        flatten_idx = next(i for i, (name, _) in enumerate(layers) if name == "flatten")
                        layers.insert(flatten_idx, ("avgpool", student_model.avgpool))
                    elif hasattr(student_model, "global_pool"):
                        flatten_idx = next(i for i, (name, _) in enumerate(layers) if name == "flatten")
                        layers.insert(flatten_idx, ("global_pool", student_model.global_pool))

            if "fc" in layer_names_to_include or "classifier" in layer_names_to_include:
                has_flatten = any(name == "flatten" for name, _ in layers)
                if not has_flatten:
                    fc_idx = next(
                        (i for i, (name, _) in enumerate(layers) if name in ["fc", "classifier"]),
                        None,
                    )
                    if fc_idx is not None:
                        has_pooling = any(name in ["avgpool", "global_pool"] for name, _ in layers)
                        if not has_pooling:
                            if hasattr(student_model, "avgpool"):
                                layers.insert(fc_idx, ("avgpool", student_model.avgpool))
                                fc_idx += 1
                            elif hasattr(student_model, "global_pool"):
                                layers.insert(fc_idx, ("global_pool", student_model.global_pool))
                                fc_idx += 1
                        layers.insert(fc_idx, ("flatten", nn.Flatten(1)))

            if "fc" in layer_names_to_include or "classifier" in layer_names_to_include:
                channel_map_resnet18 = {
                    "layer1": 64,
                    "layer2": 128,
                    "layer3": 256,
                    "layer4": 512,
                }

                if last_conv_layer and last_conv_layer in channel_map_resnet18 and hasattr(student_model, "fc"):
                    expected_features = channel_map_resnet18[last_conv_layer]
                    original_fc = student_model.fc
                    num_classes = original_fc.out_features

                    if original_fc.in_features != expected_features:
                        print("WARNING: FC layer mismatch detected!")
                        print(f"  Expected input features: {expected_features} (from {last_conv_layer})")
                        print(f"  Original FC input features: {original_fc.in_features}")
                        print(f"  Creating new FC layer: {expected_features} -> {num_classes}")
                        fc_idx = next((i for i, (name, _) in enumerate(layers) if name == "fc"), None)
                        if fc_idx is not None:
                            layers[fc_idx] = ("fc", nn.Linear(expected_features, num_classes))

        elif is_vgg:
            for layer_name in layer_names_to_include:
                if layer_name.startswith("features"):
                    features_block = student_model.features
                    if "[" in layer_name:
                        if ":" in layer_name:
                            start_idx = int(layer_name.split("[")[1].split(":")[0])
                            end_part = layer_name.split(":")[1].rstrip("]")
                            end_idx = int(end_part) if end_part else len(features_block)
                            for i in range(start_idx, end_idx):
                                layers.append((f"features[{i}]", features_block[i]))
                        else:
                            idx = int(layer_name.split("[")[1].rstrip("]"))
                            layers.append((layer_name, features_block[idx]))
                    else:
                        layers.append(("features", features_block))

                elif layer_name == "avgpool":
                    layers.append(("avgpool", student_model.avgpool))

                elif layer_name == "flatten":
                    layers.append(("flatten", nn.Flatten(1)))

                elif layer_name.startswith("classifier"):
                    classifier_block = student_model.classifier
                    if "[" in layer_name:
                        if ":" in layer_name:
                            start_idx = int(layer_name.split("[")[1].split(":")[0])
                            end_part = layer_name.split(":")[1].rstrip("]")
                            end_idx = int(end_part) if end_part else len(classifier_block)
                            for i in range(start_idx, end_idx):
                                layers.append((f"classifier[{i}]", classifier_block[i]))
                        else:
                            idx = int(layer_name.split("[")[1].rstrip("]"))
                            layers.append((layer_name, classifier_block[idx]))
                    else:
                        layers.append(("classifier", classifier_block))

            has_avgpool = any("avgpool" in name for name, _ in layers)
            has_flatten = any("flatten" in name for name, _ in layers)
            has_classifier = any("classifier" in name for name, _ in layers)

            if has_classifier and not has_flatten:
                classifier_idx = next(i for i, (name, _) in enumerate(layers) if "classifier" in name)
                if not has_avgpool:
                    layers.insert(classifier_idx, ("avgpool", student_model.avgpool))
                    classifier_idx += 1
                layers.insert(classifier_idx, ("flatten", nn.Flatten(1)))

            if has_classifier:
                expected_features = 512 * 7 * 7
                classifier_block = student_model.classifier
                for i, layer in enumerate(classifier_block):
                    if isinstance(layer, nn.Linear):
                        original_in_features = layer.in_features
                        if original_in_features != expected_features:
                            print("WARNING: VGG classifier mismatch detected!")
                            print(f"  Expected input features: {expected_features}")
                            print(f"  Original classifier input features: {original_in_features}")
                            print("  Rebuilding classifier with correct dimensions")

                            new_classifier_layers = []
                            for j, orig_layer in enumerate(classifier_block):
                                if j == 0 and isinstance(orig_layer, nn.Linear):
                                    new_classifier_layers.append(
                                        nn.Linear(expected_features, orig_layer.out_features)
                                    )
                                else:
                                    new_classifier_layers.append(orig_layer)

                            classifier_idx = next(
                                (idx for idx, (name, _) in enumerate(layers) if "classifier" in name),
                                None,
                            )
                            if classifier_idx is not None:
                                layers[classifier_idx] = ("classifier", nn.Sequential(*new_classifier_layers))
                        break

        else:
            raise ValueError("Unsupported architecture (not ResNet, VGG, or ViT)")

        print_layers = [name for name, _ in layers]
        print(f"Student trunk layers (ONLY specified): {print_layers}")
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        student_logits, sae_loss = self.hybrid_model(x)
        return student_logits, sae_loss
