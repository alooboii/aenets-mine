import torch
import torch.nn as nn
import torchvision.models as models

class TeacherModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 100, weights_path: str = None):
        super().__init__()
        self.model = self._build_model(model_name, pretrained=True, num_classes=num_classes)
        if weights_path:
            self.load_teacher_weights(weights_path)

    def _build_model(self, model_name, pretrained, num_classes):
        # Map model_name to constructors and weight enums
        if model_name.startswith('resnet'):
            num = model_name[len('resnet'):]               # e.g. "50"
            weights_enum = getattr(models, f"ResNet{num}_Weights")
            weights = weights_enum.DEFAULT if pretrained else None
            model = getattr(models, model_name)(weights=weights)
            # Replace final head
            if num_classes != model.fc.out_features:
                model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif model_name.startswith('vgg'):
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

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model

    def load_teacher_weights(self, weights_path):
        sd = torch.load(weights_path, weights_only=True)
        self.model.load_state_dict(sd, strict=False)
        print(f"[Teacher] Loaded weights from {weights_path}")

    def forward(self, x):
        return self.model(x)


class StudentModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 100, weights_path: str = None):
        super().__init__()
        self.model = self._build_model(model_name, pretrained=False, num_classes=num_classes)
        if weights_path:
            self.load_student_weights(weights_path)

    def _build_model(self, model_name, pretrained, num_classes):
        if model_name.startswith('resnet'):
            model = getattr(models, model_name)(weights=None)
            if num_classes != model.fc.out_features:
                model.fc = nn.Linear(model.fc.in_features, num_classes)

        elif model_name.startswith('vgg'):
            model = getattr(models, model_name)(weights=None)
            if num_classes != model.classifier[-1].out_features:
                layers = list(model.classifier)
                for i in reversed(range(len(layers))):
                    if isinstance(layers[i], nn.Linear):
                        in_feats = layers[i].in_features
                        layers[i] = nn.Linear(in_feats, num_classes)
                        break
                model.classifier = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return model

    def load_student_weights(self, weights_path):
        sd = torch.load(weights_path, weights_only=True)
        self.model.load_state_dict(sd, strict=False)
        print(f"[Student] Loaded weights from {weights_path}")

    def forward(self, x):
        return self.model(x)