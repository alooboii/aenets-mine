import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder (SAE) with 2D Convolutions.
    
    Args:
        input_channels (int): Number of channels in teacher features.
        student_channels (int): Latent representation size.
        lambda_sparsity (float): Weight for sparsity regularization (L1 penalty).
    """
    def __init__(self, teacher_channels, student_channels, lambda_sparsity=1e-4):
        super(SparseAutoencoder, self).__init__()
        self.lambda_sparsity = lambda_sparsity

        # Encoder: Reduces input to a lower-dimensional latent space
        self.encoder = nn.Sequential(
            nn.Conv2d(teacher_channels, student_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )

        # Decoder: Reconstructs the input from the latent space
        self.decoder = nn.Sequential(
            nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )

        self.latent = None  # Will store the encoder's output

        # Register a hook on the last layer of encoder (after ReLU)
        self.encoder[-1].register_forward_hook(self._save_latent_hook)

    def _save_latent_hook(self, module, input, output):
        self.latent = output.detach()  # Save for analysis, detach from graph

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        # Compute losses
        sparsity_loss = self.lambda_sparsity * torch.sum(torch.abs(latent))
        reconstruction_loss = F.mse_loss(reconstruction, x)
        
        total_loss = reconstruction_loss + sparsity_loss
        
        return reconstruction, latent, total_loss
