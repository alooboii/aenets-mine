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
    def __init__(
        self,
        teacher_channels,
        student_channels,
        lambda_sparsity=1e-4,
        adapter_type="conv2d",
    ):
        super(SparseAutoencoder, self).__init__()
        self.lambda_sparsity = lambda_sparsity
        self.adapter_type = adapter_type

        if adapter_type == "conv2d":
            # Encoder: Reduces input to a lower-dimensional latent space
            self.encoder = nn.Sequential(
                nn.Conv2d(
                    teacher_channels,
                    student_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
            )

            # Decoder: Reconstructs the input from the latent space
            self.decoder = nn.Sequential(
                nn.Conv2d(
                    student_channels,
                    teacher_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.ReLU(inplace=True),
            )
        elif adapter_type == "token_linear":
            self.encoder = nn.Sequential(
                nn.Linear(teacher_channels, student_channels, bias=False),
                nn.ReLU(inplace=True),
            )
            self.decoder = nn.Sequential(
                nn.Linear(student_channels, teacher_channels, bias=False),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError(
                f"Unsupported adapter_type='{adapter_type}'. "
                "Expected one of {'conv2d', 'token_linear'}."
            )

        self.latent = None  # Will store the encoder's output

        # Register a hook on the last layer of encoder (after ReLU)
        self.encoder[-1].register_forward_hook(self._save_latent_hook)

    def _save_latent_hook(self, module, input, output):
        self.latent = output.detach()  # Save for analysis, detach from graph

    def forward(self, x):
        if self.adapter_type == "conv2d":
            if x.dim() != 4:
                raise ValueError(
                    f"conv2d adapter expects 4D input [B,C,H,W], got shape {tuple(x.shape)}."
                )
        elif self.adapter_type == "token_linear":
            if x.dim() != 3:
                raise ValueError(
                    f"token_linear adapter expects 3D input [B,S,D], got shape {tuple(x.shape)}."
                )

        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        # Compute losses
        sparsity_loss = self.lambda_sparsity * torch.sum(torch.abs(latent))
        reconstruction_loss = F.mse_loss(reconstruction, x)
        
        total_loss = reconstruction_loss + sparsity_loss
        
        return reconstruction, latent, total_loss
