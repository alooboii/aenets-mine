import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from utils import get_module, count_params
from autoencoders import SparseAutoencoder


class FeatureHooks:
    """
    Helper class to extract intermediate features from a network using forward hooks.
    
    Args:
        named_layers (list of tuples): List of tuples in the form (layer_name, layer_module)
    """
    def __init__(self, named_layers):
        self.features = OrderedDict()
        self.hooks = []
        
        def hook_fn(name):
            def _hook(module, input, output):
                self.features[name] = output
            return _hook
        
        # Register a forward hook for each named layer.
        for name, layer in named_layers:
            self.hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    def clear(self):
        """Clears the stored features."""
        self.features.clear()
        
    def remove(self):
        """Removes all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

class HintLoss(nn.Module):
    """
    Computes an MSE loss between the teacher's features and the student's features,
    with optional adaptation of the student's features to match the teacher's dimensions.
    
    Args:
        teacher_channels (int): Number of channels in the teacher's feature map.
        student_channels (int): Number of channels in the student's feature map.
        adapter (str): Specifies whether to attach the adapter on the student or the teacher.
    """
    def __init__(self, teacher_channels, student_channels, adapter, sparsity):
        super(HintLoss, self).__init__()

        self.adapter = adapter
        self.teacher_channels = teacher_channels
        self.student_channels = student_channels

        if teacher_channels == student_channels:
            self.adaptation = nn.Identity()
            return
        
        if adapter == 'student':
            self.adaptation = nn.Conv2d(
                student_channels, 
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ) 
        elif adapter == 'teacher':
            self.adaptation = nn.Conv2d(
                teacher_channels,
                student_channels, 
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ) 
        elif adapter == 'teacher_SAE':
            self.adaptation = SparseAutoencoder(teacher_channels, student_channels, sparsity)
        elif adapter == 'student_SAE':
            self.adaptation = SparseAutoencoder(student_channels, teacher_channels, sparsity)
        else:
            raise ValueError(f'{adapter} is not valid. Choose from student, teacher, or SAE.')

        print(f'{adapter} adapter has {count_params(self.adaptation)} params...\n')

    def forward(self, teacher_features, student_features):
        # Adapt student's features.
        if self.adapter == 'student':
            adapted_student = self.adaptation(student_features)
            if teacher_features.shape != adapted_student.shape:
                adapted_student = F.interpolate(
                    adapted_student,
                    size=teacher_features.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            return F.mse_loss(adapted_student, teacher_features), None
        
        # Adapt teacher's features.
        elif self.adapter == 'teacher':
            adapted_teacher = self.adaptation(teacher_features)
            if student_features.shape != adapted_teacher.shape:
                adapted_teacher = F.interpolate(
                    adapted_teacher,
                    size=student_features.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            return F.mse_loss(adapted_teacher, student_features), None
        
        elif self.adapter == 'teacher_SAE':
            teacher_reconstructed, adapted_teacher, sae_loss  = self.adaptation(teacher_features)
            if student_features.shape != adapted_teacher.shape:
                adapted_teacher = F.interpolate(
                    adapted_teacher,
                    size=student_features.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            return F.mse_loss(adapted_teacher, student_features), sae_loss
        
        elif self.adapter == 'student_SAE':
            student_reconstructed, adapted_student, sae_loss  = self.adaptation(student_features)
            if teacher_features.shape != adapted_student.shape:
                adapted_student = F.interpolate(
                    adapted_student,
                    size=teacher_features.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )

            return F.mse_loss(adapted_student, teacher_features), sae_loss
    
class FitNet(nn.Module):
    """
    FitNet for knowledge distillation. This module trains a student network using
    hints from intermediate feature maps of a teacher network.
    
    The constructor expects a tuple containing two lists:
      - teacher_layers: A list of tuples (teacher_layer_name, teacher_channels)
      - student_layers: A list of tuples (student_layer_name, student_channels)
    
    Example:
        teacher_layers = [('layer3', 256)]
        student_layers = [('layer2', 128)]
        model = FitNet(teacher, student, (teacher_layers, student_layers))
    
    Args:
        teacher (nn.Module): Pretrained teacher network.
        student (nn.Module): Student network to be trained.
        layer_pairs (tuple): Tuple of two lists:
                             (teacher_layers, student_layers)
                             where each list contains tuples (layer_name, num_channels).
    """
    def __init__(self, teacher, student, layer_pairs, adapter, sparsity):
        super(FitNet, self).__init__()
        self.teacher = teacher
        self.student = student

        # Unpack the layer pair information.
        self.teacher_layers, self.student_layers = layer_pairs
        
        # Register hooks to capture intermediate features.
        self.teacher_hooks = FeatureHooks([
            (name, get_module(self.teacher.model, name)) for name, _ in self.teacher_layers
        ])
        self.student_hooks = FeatureHooks([
            (name, get_module(self.student.model, name)) for name, _ in self.student_layers
        ])
        
        # Determine the device and move hint loss modules accordingly.
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Create a HintLoss criterion for each paired layer.
        self.hint_criterions = nn.ModuleList([
            HintLoss(t_channels, s_channels, adapter, sparsity) 
            for (_, t_channels), (_, s_channels) in zip(self.teacher_layers, self.student_layers)
        ]).to(self.device)

    def forward(self, x):
        """
        Forward pass that computes both teacher and student outputs and accumulates
        the hint (distillation) loss from each intermediate layer pair.
        
        Args:
            x (Tensor): Input tensor.
            
        Returns:
            tuple: (teacher_logits, student_logits, total_hint_loss)
        """
        # Forward pass through teacher and student networks.
        teacher_logits = self.teacher(x)
        student_logits = self.student(x)

        # Accumulate hint loss over all paired layers.
        total_hint_loss = 0.0
        for idx, criterion in enumerate(self.hint_criterions):
            
            t_name = self.teacher_layers[idx][0]
            s_name = self.student_layers[idx][0]
            
            teacher_feature = self.teacher_hooks.features.get(t_name)
            student_feature = self.student_hooks.features.get(s_name)

            if teacher_feature is None or student_feature is None:
                raise ValueError(f"Missing features for layer pair: {t_name} and {s_name}")
            
            hint_loss, sae_loss = criterion(teacher_feature, student_feature)
            total_hint_loss += hint_loss

        return teacher_logits, student_logits, total_hint_loss, sae_loss

