import torch
import torch.nn as nn
import torchvision.models as models

from boomerang_utils import get_vit_encoder_layers, infer_model_family, is_vit_backbone, set_vit_encoder_layers


_VIT_WEIGHT_ENUMS = {
    "vit_b_16": models.ViT_B_16_Weights,
    "vit_b_32": models.ViT_B_32_Weights,
    "vit_l_16": models.ViT_L_16_Weights,
    "vit_l_32": models.ViT_L_32_Weights,
}
if hasattr(models, "ViT_H_14_Weights"):
    _VIT_WEIGHT_ENUMS["vit_h_14"] = models.ViT_H_14_Weights


def _safe_torch_load(path: str):
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def _extract_state_dict(obj):
    if not isinstance(obj, dict):
        raise ValueError(f"Checkpoint must be a dict-like object, got {type(obj)}")
    if all(isinstance(k, str) for k in obj.keys()):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
    return obj


class TeacherModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 100,
        weights_path: str = None,
        pretrained: bool = False,
        model_family: str = "auto",
        num_layers: int = None,
    ):
        super().__init__()
        self.model = self._build_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            model_family=model_family,
            num_layers=num_layers,
        )
        if weights_path:
            self.load_teacher_weights(weights_path)

    def _build_model(self, model_name, pretrained, num_classes, model_family, num_layers):
        family = infer_model_family(model_name) if model_family == "auto" else model_family
        # Map model_name to constructors and weight enums
        if family == "resnet" or model_name.startswith('resnet'):
            num = model_name[len('resnet'):]               # e.g. "50"
            weights_enum = getattr(models, f"ResNet{num}_Weights")
            weights = weights_enum.DEFAULT if pretrained else None
            model = getattr(models, model_name)(weights=weights)
            # Replace final head
            if num_classes != model.fc.out_features:
                model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif family == "vgg" or model_name.startswith('vgg'):
            weights_enum = getattr(models, f"{model_name.upper()}_Weights")
            weights = weights_enum.DEFAULT if pretrained else None
            model = getattr(models, model_name)(weights=weights)
            # Replace final head
            if num_classes != model.classifier[-1].out_features:
                layers = list(model.classifier)
                for i in reversed(range(len(layers))):
                    if isinstance(layers[i], nn.Linear):
                        in_feats = layers[i].in_features
                        layers[i] = nn.Linear(in_feats, num_classes)
                        break
                model.classifier = nn.Sequential(*layers)
        elif family == "vit" or model_name.startswith("vit"):
            if model_name not in _VIT_WEIGHT_ENUMS:
                raise ValueError(
                    f"Unsupported ViT model '{model_name}'. Supported: {sorted(_VIT_WEIGHT_ENUMS)}"
                )
            weights_enum = _VIT_WEIGHT_ENUMS[model_name]
            weights = weights_enum.DEFAULT if pretrained else None
            model = getattr(models, model_name)(weights=weights)
            if not is_vit_backbone(model):
                raise ValueError(f"Model '{model_name}' is not a torchvision ViT backbone.")

            if hasattr(model, "heads") and hasattr(model.heads, "head"):
                in_features = model.heads.head.in_features
                if model.heads.head.out_features != num_classes:
                    model.heads.head = nn.Linear(in_features, num_classes)

            if num_layers is not None:
                if num_layers <= 0:
                    raise ValueError("num_layers must be > 0 for ViT backbones.")
                current_layers = get_vit_encoder_layers(model)
                if num_layers > len(current_layers):
                    raise ValueError(
                        f"Requested num_layers={num_layers} exceeds teacher depth={len(current_layers)}."
                    )
                if num_layers < len(current_layers):
                    set_vit_encoder_layers(model, current_layers[:num_layers])

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model

    def load_teacher_weights(self, weights_path):
        raw = _safe_torch_load(weights_path)
        sd = _extract_state_dict(raw)
        msg = self.model.load_state_dict(sd, strict=False)
        if msg.missing_keys or msg.unexpected_keys:
            print(
                "[Teacher] Warning: non-strict load completed with "
                f"{len(msg.missing_keys)} missing and {len(msg.unexpected_keys)} unexpected keys."
            )
        print(f"[Teacher] Loaded weights from {weights_path}")

    def forward(self, x):
        return self.model(x)


class StudentModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 100,
        weights_path: str = None,
        model_family: str = "auto",
        num_layers: int = None,
    ):
        super().__init__()
        self.model = self._build_model(
            model_name,
            pretrained=False,
            num_classes=num_classes,
            model_family=model_family,
            num_layers=num_layers,
        )
        if weights_path:
            self.load_student_weights(weights_path)

    def _build_model(self, model_name, pretrained, num_classes, model_family, num_layers):
        family = infer_model_family(model_name) if model_family == "auto" else model_family

        if family == "resnet" or model_name.startswith('resnet'):
            model = getattr(models, model_name)(weights=None)
            if num_classes != model.fc.out_features:
                model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif family == "vgg" or model_name.startswith('vgg'):
            model = getattr(models, model_name)(weights=None)
            if num_classes != model.classifier[-1].out_features:
                layers = list(model.classifier)
                for i in reversed(range(len(layers))):
                    if isinstance(layers[i], nn.Linear):
                        in_feats = layers[i].in_features
                        layers[i] = nn.Linear(in_feats, num_classes)
                        break
                model.classifier = nn.Sequential(*layers)
        elif family == "vit" or model_name.startswith("vit"):
            if model_name not in _VIT_WEIGHT_ENUMS:
                raise ValueError(
                    f"Unsupported ViT model '{model_name}'. Supported: {sorted(_VIT_WEIGHT_ENUMS)}"
                )
            model = getattr(models, model_name)(weights=None)
            if not is_vit_backbone(model):
                raise ValueError(f"Model '{model_name}' is not a torchvision ViT backbone.")

            if hasattr(model, "heads") and hasattr(model.heads, "head"):
                in_features = model.heads.head.in_features
                if model.heads.head.out_features != num_classes:
                    model.heads.head = nn.Linear(in_features, num_classes)

            if num_layers is not None:
                if num_layers <= 0:
                    raise ValueError("num_layers must be > 0 for ViT backbones.")
                current_layers = get_vit_encoder_layers(model)
                if num_layers > len(current_layers):
                    raise ValueError(
                        f"Requested num_layers={num_layers} exceeds model depth={len(current_layers)}."
                    )
                if num_layers < len(current_layers):
                    set_vit_encoder_layers(model, current_layers[:num_layers])
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model

    def load_student_weights(self, weights_path):
        raw = _safe_torch_load(weights_path)
        sd = _extract_state_dict(raw)
        msg = self.model.load_state_dict(sd, strict=False)
        if msg.missing_keys or msg.unexpected_keys:
            print(
                "[Student] Warning: non-strict load completed with "
                f"{len(msg.missing_keys)} missing and {len(msg.unexpected_keys)} unexpected keys."
            )
        print(f"[Student] Loaded weights from {weights_path}")

    def forward(self, x):
        return self.model(x)
