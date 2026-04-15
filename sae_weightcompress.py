import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils import get_module, get_weight_shape
from autoencoders import SparseAutoencoder
from weight_compressor import create_weight_compressor


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


class SAEWeightCompressor(nn.Module):
    """
    Multi-layer SAE-based feature injection for knowledge distillation with weight compression.

    For each specified teacher->student layer pair, extracts teacher features,
    encodes them to student dimensions via a SparseAutoencoder, and injects
    the resulting latent into the corresponding student layer during its forward pass.
    
    Additionally, compresses teacher layer weights to match student layer dimensions
    and replaces student weights before the forward pass.
    
    OPTIMIZED: Bypasses early student layers whose outputs would be discarded anyway.

    Args:
        teacher (nn.Module): Pretrained teacher network with attribute `.model`.
        student (nn.Module): Student network with attribute `.model`.
        teacher_layer_names (list[str]): Names of layers in teacher to hook.
        teacher_channels (list[int]): Channel counts for each teacher layer.
        student_layer_names (list[str]): Names of student layers to replace.
        student_channels (list[int]): Channel counts expected at student layers.
        sparsity (float): Sparsity parameter for the SparseAutoencoder.
        teacher_weight_layers (list[str], optional): Names of teacher layers whose weights to compress.
        student_weight_layers (list[str], optional): Names of student layers to inject compressed weights into.
    """
    def __init__(
        self,
        teacher,
        student,
        teacher_layer_names,
        teacher_channels,
        student_layer_names,
        student_channels,
        sparsity,
        teacher_weight_layers=None,
        student_weight_layers=None,
    ):
        super(SAEWeightCompressor, self).__init__()
        assert len(teacher_layer_names) == len(student_layer_names) == \
               len(teacher_channels) == len(student_channels), \
               "Teacher and student lists must be the same length."

        self.teacher = teacher
        self.student = student
        self.sparsity = sparsity

        # Set up teacher hooks
        teacher_named = [
            (name, get_module(self.teacher.model, name))
            for name in teacher_layer_names
        ]
        self.teacher_hooks = FeatureHooks(teacher_named)

        # Create SAE adapters for each layer pair
        self.sae_adapters = nn.ModuleList([
            SparseAutoencoder(t_ch, s_ch, sparsity)
            for t_ch, s_ch in zip(teacher_channels, student_channels)
        ])

        # Get student modules to hook into
        self.student_modules = [
            get_module(self.student.model, name)
            for name in student_layer_names
        ]

        # Save layer names for bookkeeping
        self.teacher_layer_names = teacher_layer_names
        self.student_layer_names = student_layer_names
        
        # Find the earliest injection layer and bypass everything before it
        self.earliest_injection_layer = self._find_earliest_layer(student_layer_names)
        
        # ===== WEIGHT COMPRESSION SETUP =====
        self.teacher_weight_layers = teacher_weight_layers
        self.student_weight_layers = student_weight_layers
        self.weight_compressors = nn.ModuleList()
        self.teacher_weight_modules = []
        self.student_weight_modules = []
        
        if teacher_weight_layers is not None and student_weight_layers is not None:
            assert len(teacher_weight_layers) == len(student_weight_layers), \
                   "Teacher and student weight layer lists must be the same length."
            
            print(f"\n{'='*60}")
            print("Weight Compression Setup")
            print(f"{'='*60}")
            
            for t_weight_layer, s_weight_layer in zip(teacher_weight_layers, student_weight_layers):
                # Get the actual modules
                t_module = get_module(self.teacher.model, t_weight_layer)
                s_module = get_module(self.student.model, s_weight_layer)
                
                self.teacher_weight_modules.append(t_module)
                self.student_weight_modules.append(s_module)
                
                # Get weight shapes
                t_shape = get_weight_shape(self.teacher.model, t_weight_layer)
                s_shape = get_weight_shape(self.student.model, s_weight_layer)
                
                # Create weight compressor
                compressor = create_weight_compressor(t_shape, s_shape, compressor_type='auto')
                self.weight_compressors.append(compressor)
                
                print(f"Teacher Layer: {t_weight_layer} | Shape: {t_shape}")
                print(f"Student Layer: {s_weight_layer} | Shape: {s_shape}")
                print(f"Compressor Type: {type(compressor).__name__}")
                print(f"{'-'*60}")
        
        print(f"\n{'='*60}")
        print("SAEInjection: Optimized Architecture")
        print(f"{'='*60}")
        print(f"Injection layers: {student_layer_names}")
        print(f"Earliest injection: {self.earliest_injection_layer}")
        print(f"\nOptimization: Bypassing student layers before {self.earliest_injection_layer}")
        print(f"{'='*60}\n")

    def _find_earliest_layer(self, layer_names):
        """
        Find the earliest layer among injection points.
        For ResNet-style: layer1 < layer2 < layer3 < layer4
        """
        layer_order = ['layer1', 'layer2', 'layer3', 'layer4']
        earliest_idx = float('inf')
        earliest_name = None
        
        for name in layer_names:
            layer_prefix = name.split('[')[0]  # Handle 'layer2[0]' -> 'layer2'
            if layer_prefix in layer_order:
                idx = layer_order.index(layer_prefix)
                if idx < earliest_idx:
                    earliest_idx = idx
                    earliest_name = layer_prefix
        
        return earliest_name if earliest_name else layer_names[0]

    def _compress_and_inject_weights(self):
        """
        Compress teacher weights and inject them into student layers.
        This is called before each forward pass.
        """
        if not self.teacher_weight_modules:
            return
        
        with torch.no_grad():
            for t_module, s_module, compressor in zip(
                self.teacher_weight_modules, 
                self.student_weight_modules, 
                self.weight_compressors
            ):
                # Get teacher weights
                teacher_weights = t_module.weight.data
                
                # Compress weights
                compressed_weights, _ = compressor(teacher_weights)
                
                # Inject compressed weights into student module
                s_module.weight.data.copy_(compressed_weights)

    def _forward_student_from_injection(self, latents_dict):
        """
        Forward through student starting from the earliest injection layer.
        
        Args:
            latents_dict: Dictionary mapping layer names to SAE latents to inject
        """
        model = self.student.model
        
        # Start with the first latent (assumes single injection point for simplicity)
        # For multi-layer injection, you'd need more sophisticated logic
        x = list(latents_dict.values())[0]
        
        # Map of layer progression for ResNet-style architectures
        layer_order = ['layer1', 'layer2', 'layer3', 'layer4']
        
        # Find current layer position and continue from next layer
        earliest_layer = self.earliest_injection_layer
        try:
            if earliest_layer in layer_order:
                current_idx = layer_order.index(earliest_layer)
                # Continue from next layer
                for next_layer_name in layer_order[current_idx + 1:]:
                    if hasattr(model, next_layer_name):
                        # Check if this layer needs injection
                        if next_layer_name in [name.split('[')[0] for name in self.student_layer_names]:
                            # Replace with injected latent if available
                            for layer_name, latent in latents_dict.items():
                                if layer_name.startswith(next_layer_name):
                                    x = latent
                                    break
                        else:
                            # Normal forward pass
                            x = getattr(model, next_layer_name)(x)
        except Exception as e:
            print(f"Warning: Could not parse layer structure: {e}")
            # Fallback: try to use all remaining major layers
            for layer_name in layer_order:
                if hasattr(model, layer_name):
                    try:
                        x = getattr(model, layer_name)(x)
                    except:
                        pass
        
        # Final pooling and classifier
        if hasattr(model, 'avgpool'):
            x = model.avgpool(x)
        elif hasattr(model, 'global_pool'):
            x = model.global_pool(x)
        
        x = torch.flatten(x, 1)
        
        if hasattr(model, 'fc'):
            x = model.fc(x)
        elif hasattr(model, 'classifier'):
            x = model.classifier(x)
        
        return x

    def forward(self, x):
        # ===== WEIGHT COMPRESSION: Inject compressed weights BEFORE forward pass =====
        self._compress_and_inject_weights()
        
        # Run teacher and collect features
        teacher_logits = self.teacher(x)
        # Capture all teacher features
        features = [self.teacher_hooks.features.get(name) for name in self.teacher_layer_names]
        # Clear for next forward
        self.teacher_hooks.clear()

        # Compute latents and SAE losses
        latents = []
        total_sae_loss = 0.0
        for feat, adapter in zip(features, self.sae_adapters):
            if feat is None:
                raise ValueError("Missing teacher feature. Ensure correct layer names.")
            recon, latent, sae_loss = adapter(feat)
            latents.append(latent)
            # SAE loss could be a tensor
            total_sae_loss = total_sae_loss + sae_loss if sae_loss is not None else total_sae_loss

        # Create dictionary of layer names to latents
        latents_dict = {name: lat for name, lat in zip(self.student_layer_names, latents)}
        
        # Forward through student from injection point onwards
        student_logits = self._forward_student_from_injection(latents_dict)

        return teacher_logits, student_logits, total_sae_loss