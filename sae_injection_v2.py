import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from autoencoders import SparseAutoencoder

class FeatureHooks:
    """
    Helper class to extract intermediate features from a network using forward hooks.
    """
    def __init__(self, named_layers):
        self.features = OrderedDict()
        self.hooks = []
        
        def hook_fn(name):
            def _hook(module, input, output):
                self.features[name] = output
            return _hook
        
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


class HybridModel(nn.Module):
    """
    Hybrid model combining partial teacher (up to hint layer) + SAE + partial student (from specified layers).
    
    Args:
        teacher_trunk (nn.Module): Teacher layers up to and including hint layer
        sae_adapter (nn.Module): Sparse autoencoder for feature translation
        student_trunk (nn.Module): Student layers (only specified layers included)
        freeze_teacher (bool): Whether to freeze teacher parameters
    """
    def __init__(self, teacher_trunk, sae_adapter, student_trunk, freeze_teacher):
        super(HybridModel, self).__init__()
        
        self.teacher_trunk = teacher_trunk
        self.sae_adapter = sae_adapter
        self.student_trunk = student_trunk
        
        # Freeze teacher parameters if requested
        if freeze_teacher:
            print(f"Teacher trunk parameters frozen.")
            for param in self.teacher_trunk.parameters():
                param.requires_grad = False
            self.teacher_trunk.eval()
        else:
            for param in self.teacher_trunk.parameters():
                param.requires_grad = True
        
        self.freeze_teacher = freeze_teacher
    
    def forward(self, x):
        """
        Forward pass through hybrid architecture.
        
        Returns:
            student_logits: Final output from student trunk
            sae_loss: Reconstruction + sparsity loss from SAE
        """
        # Forward through teacher trunk (frozen or trainable)
        if self.freeze_teacher:
            with torch.no_grad():
                teacher_features = self.teacher_trunk(x)
        else:
            teacher_features = self.teacher_trunk(x)
        
        # Pass through SAE to translate to student dimensions
        recon, latent, sae_loss = self.sae_adapter(teacher_features)
        
        # Forward through student trunk starting with SAE latent
        student_logits = self.student_trunk(latent)
        
        return student_logits, sae_loss


class SAEInjection(nn.Module):
    """
    Single-layer SAE-based feature injection for knowledge distillation.
    
    Creates a hybrid model: Teacher (up to hint layer) -> SAE -> Student (specified layers only).
    The teacher trunk extracts features up to the hint layer, SAE translates them to student
    dimensions, and the student trunk processes them using only the explicitly specified layers.

    Args:
        teacher (nn.Module): Pretrained teacher network with attribute `.model`.
        student (nn.Module): Student network with attribute `.model`.
        teacher_layer_name (str): Name of hint layer in teacher (last layer to use).
        teacher_channels (int): Channel count at teacher hint layer.
        student_layer_names (list[str]): List of student layer names to include in trunk.
        student_channels (int): Channel count at first student layer (for SAE input).
        sparsity (float): Sparsity parameter for the SparseAutoencoder.
        freeze_teacher (bool): Whether to freeze teacher trunk parameters (default: True).
    """
    def __init__(
        self,
        teacher,
        student,
        teacher_layer_name,
        teacher_channels,
        student_layer_names,
        student_channels,
        sparsity,
        freeze_teacher=True,
    ):
        super(SAEInjection, self).__init__()
        
        self.teacher = teacher
        self.student = student
        self.sparsity = sparsity
        self.teacher_layer_name = teacher_layer_name
        self.student_layer_names = student_layer_names
        self.freeze_teacher = freeze_teacher
        
        # Create the hybrid model
        self.hybrid_model = self._create_hybrid_model(
            teacher=teacher,
            student=student,
            teacher_layer_name=teacher_layer_name,
            teacher_channels=teacher_channels,
            student_layer_names=student_layer_names,
            student_channels=student_channels,
            sparsity=sparsity,
            freeze_teacher=freeze_teacher
        )
        
        print(f"\n{'='*60}")
        print("SAEInjection: Hybrid Architecture Created")
        print(f"{'='*60}")
        print(f"Teacher hint layer: {teacher_layer_name} (channels: {teacher_channels})")
        print(f"Student layers: {student_layer_names}")
        print(f"Student input channels (for SAE): {student_channels}")
        print(f"Teacher frozen: {freeze_teacher}")
        print(f"Sparsity coefficient: {sparsity}")
        print(f"\nArchitecture: Teacher[...→{teacher_layer_name}] → SAE → Student{student_layer_names}")
        print(f"\tTotal params [hybrid model]: {sum(p.numel() for p in self.hybrid_model.parameters()) * 1e-6:.2f}M")
        print(f"\tTotal params [teacher trunk]: {sum(p.numel() for p in self.hybrid_model.teacher_trunk.parameters()) * 1e-6:.2f}M")
        print(f"\tTotal params [student trunk]: {sum(p.numel() for p in self.hybrid_model.student_trunk.parameters()) * 1e-6:.2f}M")
        print(f"\tTotal params [SAE]: {sum(p.numel() for p in self.hybrid_model.sae_adapter.parameters()) * 1e-6:.2f}M")
        print(f"{'='*60}\n")

    def _create_hybrid_model(
        self,
        teacher,
        student,
        teacher_layer_name,
        teacher_channels,
        student_layer_names,
        student_channels,
        sparsity,
        freeze_teacher
    ):
        """
        Creates and returns the hybrid model combining:
        - Teacher trunk (up to and including hint layer)
        - SAE adapter (feature translation)
        - Student trunk (only specified layers)
        
        Args:
            teacher: Full teacher model
            student: Full student model
            teacher_layer_name: Hint layer name
            teacher_channels: Channels at hint layer
            student_layer_names: List of student layer names to include
            student_channels: Channels at first student layer (SAE input)
            sparsity: SAE sparsity parameter
            freeze_teacher: Whether to freeze teacher parameters
            
        Returns:
            HybridModel instance
        """
        # Build teacher trunk (up to and including hint layer)
        teacher_trunk = self._build_teacher_trunk(teacher.model, teacher_layer_name)
        
        # Build student trunk (only specified layers)
        student_trunk = self._build_student_trunk(student.model, student_layer_names)
        
        # Create SAE adapter
        sae_adapter = SparseAutoencoder(teacher_channels, student_channels, sparsity)
        
        # Combine into hybrid model
        hybrid_model = HybridModel(
            teacher_trunk=teacher_trunk,
            sae_adapter=sae_adapter,
            student_trunk=student_trunk,
            freeze_teacher=freeze_teacher
        )
        
        return hybrid_model
    
    def _build_teacher_trunk(self, teacher_model, hint_layer_name):
        """
        Extracts teacher layers up to and including the hint layer.
        Supports both ResNet and VGG architectures.
        
        Args:
            teacher_model: Teacher's .model attribute
            hint_layer_name: Name of hint layer (e.g., 'layer3', 'features[10]')
            
        Returns:
            nn.Sequential containing the teacher trunk
        """
        layers = []
        
        # Detect architecture type
        is_vgg = hasattr(teacher_model, 'features') and hasattr(teacher_model, 'classifier')
        is_resnet = hasattr(teacher_model, 'layer1')
        
        if is_resnet:
            # ResNet architecture handling
            initial_layer_names = ['conv1', 'bn1', 'relu', 'maxpool']
            for name in initial_layer_names:
                if hasattr(teacher_model, name):
                    layers.append((name, getattr(teacher_model, name)))
            
            # Parse hint layer name
            hint_base = hint_layer_name.split('[')[0]
            hint_idx = None
            if '[' in hint_layer_name:
                hint_idx = int(hint_layer_name.split('[')[1].rstrip(']'))
            
            # Add layer blocks up to and including hint layer
            layer_order = ['layer1', 'layer2', 'layer3', 'layer4']
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
                    else:
                        layers.append((layer_name, layer_block))
                        
                    if layer_order.index(layer_name) >= layer_order.index(hint_base):
                        break
        
        elif is_vgg:
            # VGG architecture handling
            # Parse hint layer name (e.g., 'features[10]' or 'features')
            if hint_layer_name.startswith('features'):
                features_block = teacher_model.features
                
                if '[' in hint_layer_name:
                    # Specific index in features
                    hint_idx = int(hint_layer_name.split('[')[1].rstrip(']'))
                    for i in range(hint_idx + 1):
                        layers.append((f"features[{i}]", features_block[i]))
                else:
                    # All features
                    layers.append(('features', features_block))
            
            elif hint_layer_name.startswith('classifier'):
                # Include all features + avgpool + flatten + part of classifier
                layers.append(('features', teacher_model.features))
                layers.append(('avgpool', teacher_model.avgpool))
                layers.append(('flatten', nn.Flatten(1)))
                
                classifier_block = teacher_model.classifier
                if '[' in hint_layer_name:
                    hint_idx = int(hint_layer_name.split('[')[1].rstrip(']'))
                    for i in range(hint_idx + 1):
                        layers.append((f"classifier[{i}]", classifier_block[i]))
                else:
                    layers.append(('classifier', classifier_block))
            else:
                raise ValueError(f"Invalid VGG hint layer name: {hint_layer_name}")
        
        else:
            raise ValueError(f"Unsupported architecture (not ResNet or VGG)")
        
        return nn.Sequential(OrderedDict(layers))
    
    def _build_student_trunk(self, student_model, layer_names_to_include):
        """
        Extracts ONLY the specified student layers.
        Supports both ResNet and VGG architectures.
        """
        layers = []
        
        # Detect architecture
        is_vgg = hasattr(student_model, 'features') and hasattr(student_model, 'classifier')
        is_resnet = hasattr(student_model, 'layer1')
        
        if is_resnet:
            # ResNet handling (existing code)
            all_layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'global_pool', 'flatten', 'fc', 'classifier']
            last_conv_layer = None
            
            for layer_name in all_layer_names:
                if layer_name in layer_names_to_include:
                    if hasattr(student_model, layer_name):
                        layer = getattr(student_model, layer_name)
                        if layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                            last_conv_layer = layer_name
                        layers.append((layer_name, layer))
                    elif layer_name == 'flatten':
                        layers.append(('flatten', nn.Flatten(1)))
                    else:
                        print(f"Warning: Requested layer '{layer_name}' not found in student model")
            
            # Auto-add pooling and flatten if fc/classifier is requested
            if 'flatten' in layer_names_to_include:
                has_pooling = any(name in ['avgpool', 'global_pool'] for name, _ in layers)
                if not has_pooling:
                    if hasattr(student_model, 'avgpool'):
                        flatten_idx = next(i for i, (name, _) in enumerate(layers) if name == 'flatten')
                        layers.insert(flatten_idx, ('avgpool', student_model.avgpool))
                    elif hasattr(student_model, 'global_pool'):
                        flatten_idx = next(i for i, (name, _) in enumerate(layers) if name == 'flatten')
                        layers.insert(flatten_idx, ('global_pool', student_model.global_pool))
            
            if ('fc' in layer_names_to_include or 'classifier' in layer_names_to_include):
                has_flatten = any(name == 'flatten' for name, _ in layers)
                if not has_flatten:
                    fc_idx = next((i for i, (name, _) in enumerate(layers) if name in ['fc', 'classifier']), None)
                    if fc_idx is not None:
                        has_pooling = any(name in ['avgpool', 'global_pool'] for name, _ in layers)
                        if not has_pooling:
                            if hasattr(student_model, 'avgpool'):
                                layers.insert(fc_idx, ('avgpool', student_model.avgpool))
                                fc_idx += 1
                            elif hasattr(student_model, 'global_pool'):
                                layers.insert(fc_idx, ('global_pool', student_model.global_pool))
                                fc_idx += 1
                        layers.insert(fc_idx, ('flatten', nn.Flatten(1)))
            
            # Fix FC layer dimension mismatch
            if ('fc' in layer_names_to_include or 'classifier' in layer_names_to_include):
                channel_map_resnet18 = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
                
                if last_conv_layer and last_conv_layer in channel_map_resnet18:
                    expected_features = channel_map_resnet18[last_conv_layer]
                    
                    if hasattr(student_model, 'fc'):
                        original_fc = student_model.fc
                        num_classes = original_fc.out_features
                        
                        if original_fc.in_features != expected_features:
                            print(f"WARNING: FC layer mismatch detected!")
                            print(f"  Expected input features: {expected_features} (from {last_conv_layer})")
                            print(f"  Original FC input features: {original_fc.in_features}")
                            print(f"  Creating new FC layer: {expected_features} -> {num_classes}")
                            
                            fc_idx = next((i for i, (name, _) in enumerate(layers) if name == 'fc'), None)
                            if fc_idx is not None:
                                new_fc = nn.Linear(expected_features, num_classes)
                                layers[fc_idx] = ('fc', new_fc)
        
        elif is_vgg:
            # VGG handling
            for layer_name in layer_names_to_include:
                if layer_name.startswith('features'):
                    features_block = student_model.features
                    if '[' in layer_name:
                        # Specific index range (e.g., 'features[10:]')
                        if ':' in layer_name:
                            start_idx = int(layer_name.split('[')[1].split(':')[0])
                            end_part = layer_name.split(':')[1].rstrip(']')
                            end_idx = int(end_part) if end_part else len(features_block)
                            for i in range(start_idx, end_idx):
                                layers.append((f"features[{i}]", features_block[i]))
                        else:
                            # Single index
                            idx = int(layer_name.split('[')[1].rstrip(']'))
                            layers.append((layer_name, features_block[idx]))
                    else:
                        # All features
                        layers.append(('features', features_block))
                
                elif layer_name == 'avgpool':
                    layers.append(('avgpool', student_model.avgpool))
                
                elif layer_name == 'flatten':
                    layers.append(('flatten', nn.Flatten(1)))
                
                elif layer_name.startswith('classifier'):
                    classifier_block = student_model.classifier
                    if '[' in layer_name:
                        if ':' in layer_name:
                            start_idx = int(layer_name.split('[')[1].split(':')[0])
                            end_part = layer_name.split(':')[1].rstrip(']')
                            end_idx = int(end_part) if end_part else len(classifier_block)
                            for i in range(start_idx, end_idx):
                                layers.append((f"classifier[{i}]", classifier_block[i]))
                        else:
                            idx = int(layer_name.split('[')[1].rstrip(']'))
                            layers.append((layer_name, classifier_block[idx]))
                    else:
                        layers.append(('classifier', classifier_block))
            
            # Auto-add necessary layers for VGG
            has_avgpool = any('avgpool' in name for name, _ in layers)
            has_flatten = any('flatten' in name for name, _ in layers)
            has_classifier = any('classifier' in name for name, _ in layers)
            
            if has_classifier and not has_flatten:
                classifier_idx = next(i for i, (name, _) in enumerate(layers) if 'classifier' in name)
                if not has_avgpool:
                    layers.insert(classifier_idx, ('avgpool', student_model.avgpool))
                    classifier_idx += 1
                layers.insert(classifier_idx, ('flatten', nn.Flatten(1)))
            
            # Fix classifier dimension mismatch for VGG
            if has_classifier:
                # Determine expected features based on VGG architecture
                vgg_feature_map = {
                    'vgg11': 512, 'vgg13': 512, 'vgg16': 512, 'vgg19': 512,
                    'vgg11_bn': 512, 'vgg13_bn': 512, 'vgg16_bn': 512, 'vgg19_bn': 512
                }
                
                # Check if we can find the architecture name
                model_name = student_model.__class__.__name__.lower()
                expected_features = 512 * 7 * 7  # Default VGG flattened size
                
                # Find first Linear layer in classifier
                classifier_block = student_model.classifier
                for i, layer in enumerate(classifier_block):
                    if isinstance(layer, nn.Linear):
                        original_in_features = layer.in_features
                        num_classes = classifier_block[-1].out_features if isinstance(classifier_block[-1], nn.Linear) else layer.out_features
                        
                        if original_in_features != expected_features:
                            print(f"WARNING: VGG classifier mismatch detected!")
                            print(f"  Expected input features: {expected_features}")
                            print(f"  Original classifier input features: {original_in_features}")
                            print(f"  Rebuilding classifier with correct dimensions")
                            
                            # Rebuild classifier with correct input size
                            new_classifier_layers = []
                            for j, orig_layer in enumerate(classifier_block):
                                if j == 0 and isinstance(orig_layer, nn.Linear):
                                    new_classifier_layers.append(nn.Linear(expected_features, orig_layer.out_features))
                                else:
                                    new_classifier_layers.append(orig_layer)
                            
                            # Replace in layers list
                            classifier_idx = next((idx for idx, (name, _) in enumerate(layers) if 'classifier' in name), None)
                            if classifier_idx is not None:
                                layers[classifier_idx] = ('classifier', nn.Sequential(*new_classifier_layers))
                        break
        
        else:
            raise ValueError("Unsupported architecture (not ResNet or VGG)")
        
        print_layers = [name for name, _ in layers]
        print(f"Student trunk layers (ONLY specified): {print_layers}")
        
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor
            
        Returns:
            student_logits: Predictions from student trunk
            sae_loss: SAE reconstruction + sparsity loss
        """
        student_logits, sae_loss = self.hybrid_model(x)
        return student_logits, sae_loss