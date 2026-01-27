### COMMENTS CONTAIN WEIGHT COMPRESSION SETUP ###
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from collections import OrderedDict
# from utils import get_module, get_weight_shape
# from autoencoders import SparseAutoencoder
# from weight_compressor import create_weight_compressor


# class FeatureHooks:
#     """
#     Helper class to extract intermediate features from a network using forward hooks.
#     """
#     def __init__(self, named_layers):
#         self.features = OrderedDict()
#         self.hooks = []
        
#         def hook_fn(name):
#             def _hook(module, input, output):
#                 self.features[name] = output
#             return _hook
        
#         for name, layer in named_layers:
#             self.hooks.append(layer.register_forward_hook(hook_fn(name)))
    
#     def clear(self):
#         """Clears the stored features."""
#         self.features.clear()
        
#     def remove(self):
#         """Removes all hooks."""
#         for hook in self.hooks:
#             hook.remove()
#         self.hooks.clear()


# class SAEInjection(nn.Module):
#     """
#     Multi-layer SAE-based feature injection for knowledge distillation with weight compression.

#     For each specified teacher->student layer pair, extracts teacher features,
#     encodes them to student dimensions via a SparseAutoencoder, and injects
#     the resulting latent into the corresponding student layer during its forward pass.
    
#     Additionally, compresses teacher layer weights to match student layer dimensions
#     and replaces student weights before the forward pass.
    
#     OPTIMIZED: Bypasses early student layers whose outputs would be discarded anyway.

#     Args:
#         teacher (nn.Module): Pretrained teacher network with attribute `.model`.
#         student (nn.Module): Student network with attribute `.model`.
#         teacher_layer_names (list[str]): Names of layers in teacher to hook.
#         teacher_channels (list[int]): Channel counts for each teacher layer.
#         student_layer_names (list[str]): Names of student layers to replace.
#         student_channels (list[int]): Channel counts expected at student layers.
#         sparsity (float): Sparsity parameter for the SparseAutoencoder.
#         teacher_weight_layers (list[str], optional): Names of teacher layers whose weights to compress.
#         student_weight_layers (list[str], optional): Names of student layers to inject compressed weights into.
#     """
#     def __init__(
#         self,
#         teacher,
#         student,
#         teacher_layer_names,
#         teacher_channels,
#         student_layer_names,
#         student_channels,
#         sparsity,
#         teacher_weight_layers=None,
#         student_weight_layers=None,
#     ):
#         super(SAEInjection, self).__init__()
#         assert len(teacher_layer_names) == len(student_layer_names) == \
#                len(teacher_channels) == len(student_channels), \
#                "Teacher and student lists must be the same length."

#         self.teacher = teacher
#         self.student = student
#         self.sparsity = sparsity

#         # Set up teacher hooks
#         teacher_named = [
#             (name, get_module(self.teacher.model, name))
#             for name in teacher_layer_names
#         ]
#         self.teacher_hooks = FeatureHooks(teacher_named)

#         # Create SAE adapters for each layer pair
#         self.sae_adapters = nn.ModuleList([
#             SparseAutoencoder(t_ch, s_ch, sparsity)
#             for t_ch, s_ch in zip(teacher_channels, student_channels)
#         ])

#         # Get student modules to hook into
#         self.student_modules = [
#             get_module(self.student.model, name)
#             for name in student_layer_names
#         ]

#         # Save layer names for bookkeeping
#         self.teacher_layer_names = teacher_layer_names
#         self.student_layer_names = student_layer_names
        
#         # Find the earliest injection layer and bypass everything before it
#         self.earliest_injection_layer = self._find_earliest_layer(student_layer_names)
        
#         # ===== WEIGHT COMPRESSION SETUP =====
#         self.teacher_weight_layers = teacher_weight_layers
#         self.student_weight_layers = student_weight_layers
#         self.weight_compressors = nn.ModuleList()
#         self.teacher_weight_modules = []
#         self.student_weight_modules = []
        
#         if teacher_weight_layers is not None and student_weight_layers is not None:
#             assert len(teacher_weight_layers) == len(student_weight_layers), \
#                    "Teacher and student weight layer lists must be the same length."
            
#             print(f"\n{'='*60}")
#             print("Weight Compression Setup")
#             print(f"{'='*60}")
            
#             for t_weight_layer, s_weight_layer in zip(teacher_weight_layers, student_weight_layers):
#                 # Get the actual modules
#                 t_module = get_module(self.teacher.model, t_weight_layer)
#                 s_module = get_module(self.student.model, s_weight_layer)
                
#                 self.teacher_weight_modules.append(t_module)
#                 self.student_weight_modules.append(s_module)
                
#                 # Get weight shapes
#                 t_shape = get_weight_shape(self.teacher.model, t_weight_layer)
#                 s_shape = get_weight_shape(self.student.model, s_weight_layer)
                
#                 # Create weight compressor
#                 compressor = create_weight_compressor(t_shape, s_shape, compressor_type='auto')
#                 self.weight_compressors.append(compressor)
                
#                 print(f"Teacher Layer: {t_weight_layer} | Shape: {t_shape}")
#                 print(f"Student Layer: {s_weight_layer} | Shape: {s_shape}")
#                 print(f"Compressor Type: {type(compressor).__name__}")
#                 print(f"{'-'*60}")
        
#         print(f"\n{'='*60}")
#         print("SAEInjection: Optimized Architecture")
#         print(f"{'='*60}")
#         print(f"Injection layers: {student_layer_names}")
#         print(f"Earliest injection: {self.earliest_injection_layer}")
#         print(f"\nOptimization: Bypassing student layers before {self.earliest_injection_layer}")
#         print(f"{'='*60}\n")

#     def _find_earliest_layer(self, layer_names):
#         """
#         Find the earliest layer among injection points.
#         For ResNet-style: layer1 < layer2 < layer3 < layer4
#         """
#         layer_order = ['layer1', 'layer2', 'layer3', 'layer4']
#         earliest_idx = float('inf')
#         earliest_name = None
        
#         for name in layer_names:
#             layer_prefix = name.split('[')[0]  # Handle 'layer2[0]' -> 'layer2'
#             if layer_prefix in layer_order:
#                 idx = layer_order.index(layer_prefix)
#                 if idx < earliest_idx:
#                     earliest_idx = idx
#                     earliest_name = layer_prefix
        
#         return earliest_name if earliest_name else layer_names[0]

#     def _compress_and_inject_weights(self):
#         """
#         Compress teacher weights and inject them into student layers.
#         This is called before each forward pass.
#         """
#         if not self.teacher_weight_modules:
#             return
        
#         with torch.no_grad():
#             for t_module, s_module, compressor in zip(
#                 self.teacher_weight_modules, 
#                 self.student_weight_modules, 
#                 self.weight_compressors
#             ):
#                 # Get teacher weights
#                 teacher_weights = t_module.weight.data
                
#                 # Compress weights
#                 compressed_weights, _ = compressor(teacher_weights)
                
#                 # Inject compressed weights into student module
#                 s_module.weight.data.copy_(compressed_weights)

#     def _forward_student_from_injection(self, latents_dict):
#         """
#         Forward through student starting from the earliest injection layer.
        
#         Args:
#             latents_dict: Dictionary mapping layer names to SAE latents to inject
#         """
#         model = self.student.model
        
#         # Start with the first latent (assumes single injection point for simplicity)
#         # For multi-layer injection, you'd need more sophisticated logic
#         x = list(latents_dict.values())[0]
        
#         # Map of layer progression for ResNet-style architectures
#         layer_order = ['layer1', 'layer2', 'layer3', 'layer4']
        
#         # Find current layer position and continue from next layer
#         earliest_layer = self.earliest_injection_layer
#         try:
#             if earliest_layer in layer_order:
#                 current_idx = layer_order.index(earliest_layer)
#                 # Continue from next layer
#                 for next_layer_name in layer_order[current_idx + 1:]:
#                     if hasattr(model, next_layer_name):
#                         # Check if this layer needs injection
#                         if next_layer_name in [name.split('[')[0] for name in self.student_layer_names]:
#                             # Replace with injected latent if available
#                             for layer_name, latent in latents_dict.items():
#                                 if layer_name.startswith(next_layer_name):
#                                     x = latent
#                                     break
#                         else:
#                             # Normal forward pass
#                             x = getattr(model, next_layer_name)(x)
#         except Exception as e:
#             print(f"Warning: Could not parse layer structure: {e}")
#             # Fallback: try to use all remaining major layers
#             for layer_name in layer_order:
#                 if hasattr(model, layer_name):
#                     try:
#                         x = getattr(model, layer_name)(x)
#                     except:
#                         pass
        
#         # Final pooling and classifier
#         if hasattr(model, 'avgpool'):
#             x = model.avgpool(x)
#         elif hasattr(model, 'global_pool'):
#             x = model.global_pool(x)
        
#         x = torch.flatten(x, 1)
        
#         if hasattr(model, 'fc'):
#             x = model.fc(x)
#         elif hasattr(model, 'classifier'):
#             x = model.classifier(x)
        
#         return x

#     def forward(self, x):
#         # ===== WEIGHT COMPRESSION: Inject compressed weights BEFORE forward pass =====
#         self._compress_and_inject_weights()
        
#         # Run teacher and collect features
#         teacher_logits = self.teacher(x)
#         # Capture all teacher features
#         features = [self.teacher_hooks.features.get(name) for name in self.teacher_layer_names]
#         # Clear for next forward
#         self.teacher_hooks.clear()

#         # Compute latents and SAE losses
#         latents = []
#         total_sae_loss = 0.0
#         for feat, adapter in zip(features, self.sae_adapters):
#             if feat is None:
#                 raise ValueError("Missing teacher feature. Ensure correct layer names.")
#             recon, latent, sae_loss = adapter(feat)
#             latents.append(latent)
#             # SAE loss could be a tensor
#             total_sae_loss = total_sae_loss + sae_loss if sae_loss is not None else total_sae_loss

#         # Create dictionary of layer names to latents
#         latents_dict = {name: lat for name, lat in zip(self.student_layer_names, latents)}
        
#         # Forward through student from injection point onwards
#         student_logits = self._forward_student_from_injection(latents_dict)

#         return teacher_logits, student_logits, total_sae_loss

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
        
        Args:
            teacher_model: Teacher's .model attribute
            hint_layer_name: Name of hint layer (e.g., 'layer3', 'layer2[1]')
            
        Returns:
            nn.Sequential containing the teacher trunk
        """
        layers = []
        
        # Handle initial layers (conv1, bn1, relu, maxpool for ResNet-style)
        initial_layer_names = ['conv1', 'bn1', 'relu', 'maxpool']
        for name in initial_layer_names:
            if hasattr(teacher_model, name):
                layers.append((name, getattr(teacher_model, name)))
        
        # Parse hint layer name to determine which blocks to include
        hint_base = hint_layer_name.split('[')[0]  # 'layer3[0]' -> 'layer3'
        hint_idx = None
        if '[' in hint_layer_name:
            hint_idx = int(hint_layer_name.split('[')[1].rstrip(']'))
        
        # Add layer blocks up to and including hint layer
        layer_order = ['layer1', 'layer2', 'layer3', 'layer4']
        for layer_name in layer_order:
            if hasattr(teacher_model, layer_name):
                layer_block = getattr(teacher_model, layer_name)
                
                if layer_name == hint_base:
                    # Include up to hint_idx if specified
                    if hint_idx is not None and isinstance(layer_block, nn.Sequential):
                        for i in range(hint_idx + 1):
                            layers.append((f"{layer_name}[{i}]", layer_block[i]))
                    else:
                        layers.append((layer_name, layer_block))
                    break  # Stop after hint layer
                else:
                    layers.append((layer_name, layer_block))
                    
                # Check if we've passed the hint layer
                if layer_order.index(layer_name) >= layer_order.index(hint_base):
                    break
        
        return nn.Sequential(OrderedDict(layers))
    
    def _build_student_trunk(self, student_model, layer_names_to_include):
        """
        Extracts ONLY the specified student layers.
        """
        layers = []
        
        # Define all possible layer names in order
        all_layer_names = ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'global_pool', 'flatten', 'fc', 'classifier']
        
        # Track the last layer to determine output channels
        last_conv_layer = None
        
        for layer_name in all_layer_names:
            # Check if this layer should be included
            if layer_name in layer_names_to_include:
                # Handle standard layers
                if hasattr(student_model, layer_name):
                    layer = getattr(student_model, layer_name)
                    
                    # Track last convolutional layer for dimension calculation
                    if layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                        last_conv_layer = layer_name
                    
                    layers.append((layer_name, layer))
                # Handle special case for flatten
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
        
        # **NEW: Fix FC layer dimension mismatch**
        if ('fc' in layer_names_to_include or 'classifier' in layer_names_to_include):
            # Determine expected input features based on last conv layer
            channel_map_resnet18 = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}
            
            if last_conv_layer and last_conv_layer in channel_map_resnet18:
                expected_features = channel_map_resnet18[last_conv_layer]
                
                # Get the original FC layer
                if hasattr(student_model, 'fc'):
                    original_fc = student_model.fc
                    num_classes = original_fc.out_features
                    
                    # If dimensions don't match, create a new FC layer
                    if original_fc.in_features != expected_features:
                        print(f"WARNING: FC layer mismatch detected!")
                        print(f"  Expected input features: {expected_features} (from {last_conv_layer})")
                        print(f"  Original FC input features: {original_fc.in_features}")
                        print(f"  Creating new FC layer: {expected_features} -> {num_classes}")
                        
                        # Replace the FC layer in the layers list
                        fc_idx = next((i for i, (name, _) in enumerate(layers) if name == 'fc'), None)
                        if fc_idx is not None:
                            new_fc = nn.Linear(expected_features, num_classes)
                            layers[fc_idx] = ('fc', new_fc)
        
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