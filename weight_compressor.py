"""
Weight compressor module for AENets.
Handles compression of teacher weights to student dimensions.
Supports both convolutional (4D) and fully connected (2D) weight tensors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_weight_compressor(teacher_shape, student_shape, compressor_type='auto'):
    """
    Factory function to create appropriate weight compressor based on tensor dimensions.
    
    Args:
        teacher_shape: Shape of teacher weights (tuple)
        student_shape: Shape of student weights (tuple)
        compressor_type: 'auto', 'conv', or 'linear'
    
    Returns:
        Appropriate weight compressor instance
    """
    # Auto-detect based on dimensionality
    if compressor_type == 'auto':
        if len(teacher_shape) == 4 and len(student_shape) == 4:
            compressor_type = 'conv'
        elif len(teacher_shape) == 2 and len(student_shape) == 2:
            compressor_type = 'linear'
        else:
            raise ValueError(
                f"Cannot auto-detect compressor type for shapes "
                f"teacher={teacher_shape}, student={student_shape}. "
                f"Please specify 'conv' or 'linear' explicitly."
            )
    
    # Create appropriate compressor
    if compressor_type == 'conv':
        if len(teacher_shape) != 4 or len(student_shape) != 4:
            raise ValueError(
                f"ConvWeightCompressor requires 4D shapes, got "
                f"teacher={teacher_shape}, student={student_shape}"
            )
        return ConvWeightCompressor(teacher_shape, student_shape)
    
    elif compressor_type == 'linear':
        if len(teacher_shape) != 2 or len(student_shape) != 2:
            raise ValueError(
                f"WeightAutoencoder requires 2D shapes, got "
                f"teacher={teacher_shape}, student={student_shape}"
            )
        return WeightAutoencoder(teacher_shape, student_shape)
    
    else:
        raise ValueError(f"Unknown compressor_type: {compressor_type}")


class ConvWeightCompressor(nn.Module):
    """
    Use depthwise-separable convolutions to compress weight tensors.
    Treats the weight tensor as a 4D feature map.
    
    ONLY works with 4D convolutional weight tensors!
    """
    def __init__(self, teacher_shape, student_shape):
        super().__init__()
        
        # Validate shapes
        if len(teacher_shape) != 4 or len(student_shape) != 4:
            raise ValueError(
                f"ConvWeightCompressor requires 4D shapes (out_ch, in_ch, k_h, k_w), "
                f"got teacher={teacher_shape}, student={student_shape}. "
                f"For 2D weights (FC layers), use WeightAutoencoder instead."
            )
        
        self.t_out, self.t_in, self.t_k, _ = teacher_shape
        self.s_out, self.s_in, self.s_k, _ = student_shape
        
        self.teacher_shape = teacher_shape
        self.student_shape = student_shape
        
        # First: reduce output channels
        self.compress_out = nn.Conv1d(self.t_out, self.s_out, kernel_size=1, bias=False)
        
        # Second: reduce input channels  
        self.compress_in = nn.Conv1d(self.t_in, self.s_in, kernel_size=1, bias=False)
        
        # Spatial pooling if needed
        self.spatial_pool = nn.AdaptiveAvgPool2d((self.s_k, self.s_k)) if self.t_k != self.s_k else None
        
    def forward(self, teacher_weights):
        """
        Args:
            teacher_weights: Teacher layer weights [t_out, t_in, t_k, t_k]
        Returns:
            compressed_weights: Compressed weights matching student shape
            reconstructed_weights: Reconstructed teacher weights (for loss computation)
        """
        t_out, t_in, t_k, _ = teacher_weights.shape
        assert t_out == self.t_out and t_in == self.t_in, \
            f"Teacher shape mismatch: got {(t_out,t_in)} expected {(self.t_out,self.t_in)}"

        # Reshape to [1, t_out, t_in * t_k * t_k]
        temp = teacher_weights.view(t_out, -1).unsqueeze(0)
        # Compress output channels
        temp = self.compress_out(temp)
        temp = temp.view(self.s_out, t_in, t_k, t_k)

        # Compress input channels
        temp2 = temp.permute(1, 0, 2, 3).reshape(t_in, -1).unsqueeze(0)
        temp2 = self.compress_in(temp2)
        compressed = temp2.view(self.s_out, self.s_in, t_k, t_k)

        if self.spatial_pool:
            compressed = self.spatial_pool(compressed)

        # Reconstruction
        reconstructed = compressed

        # Upsample spatial dims if needed
        if self.spatial_pool:
            reconstructed = F.interpolate(reconstructed, size=(t_k, t_k), mode='bilinear', align_corners=False)

        # Invert compress_in
        rec = reconstructed.permute(1, 0, 2, 3).reshape(self.s_in, -1).unsqueeze(0)
        inv_weight_in = self.compress_in.weight.transpose(0, 1).contiguous()
        rec = F.conv1d(rec, inv_weight_in)
        rec = rec.view(self.s_out, t_in, t_k, t_k)

        # Invert compress_out
        rec2 = rec.view(self.s_out, -1).unsqueeze(0)
        inv_weight_out = self.compress_out.weight.transpose(0, 1).contiguous()
        rec2 = F.conv1d(rec2, inv_weight_out)
        rec2 = rec2.view(t_out, t_in, t_k, t_k)

        return compressed, rec2


class WeightAutoencoder(nn.Module):
    """
    Lightweight autoencoder for compressing teacher layer weights to match student dimensions.
    Uses weight tying and L1 sparsity penalty.
    
    Works with 2D fully connected layer weight tensors.
    """
    def __init__(self, input_shape, output_shape):
        super().__init__()
        
        # Validate shapes
        if len(input_shape) != 2 or len(output_shape) != 2:
            raise ValueError(
                f"WeightAutoencoder requires 2D shapes (out_features, in_features), "
                f"got input_shape={input_shape}, output_shape={output_shape}. "
                f"For 4D weights (Conv layers), use ConvWeightCompressor instead."
            )
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # Flatten shapes for linear layers
        self.input_dim = input_shape[0] * input_shape[1]
        self.latent_dim = output_shape[0] * output_shape[1]
        
        # Encoder: compress teacher weights to student weight dimensions
        self.encoder = nn.Linear(self.input_dim, self.latent_dim, bias=False)
        
    def forward(self, teacher_weights):
        """
        Args:
            teacher_weights: Teacher layer weights [out_features, in_features]
        Returns:
            compressed_weights: Compressed weights matching student shape
            reconstructed_weights: Reconstructed teacher weights
        """
        # Validate input shape
        if teacher_weights.shape != self.input_shape:
            raise ValueError(
                f"Expected teacher weights with shape {self.input_shape}, "
                f"got {teacher_weights.shape}"
            )
        
        # Flatten teacher weights
        batch_flat = teacher_weights.view(-1, self.input_dim)
        
        # Encode to latent (compressed weights)
        latent = self.encoder(batch_flat)
        
        # Decode with weight tying
        reconstructed_flat = F.linear(latent, self.encoder.weight.t())
        
        # Reshape outputs
        compressed_weights = latent.view(self.output_shape)
        reconstructed_weights = reconstructed_flat.view(self.input_shape)
        
        return compressed_weights, reconstructed_weights