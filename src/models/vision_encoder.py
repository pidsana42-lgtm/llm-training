import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.transformer_block import RMSNorm
from src.models.mlp import MLP

class PatchEmbedding(nn.Module):
    """
    Converts an image into a sequence of patches (Vision Tokens).
    This is the first step for Jommarn-Tiny to "see".
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, n_embed=192):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Use a convolution to create patches and project to n_embed dimension
        self.proj = nn.Conv2d(
            in_chans, n_embed, 
            kernel_size=patch_size, 
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x) # (B, n_embed, H/P, W/P)
        x = x.flatten(2) # (B, n_embed, n_patches)
        x = x.transpose(1, 2) # (B, n_patches, n_embed)
        return x

class VisionBlock(nn.Module):
    """
    A lightweight Transformer block for image features.
    Uses the same Jommarn-Tiny philosophy (RMSNorm + SwiGLU).
    """
    def __init__(self, n_head, n_embed):
        super().__init__()
        self.ln1 = RMSNorm(n_embed)
        self.attn = nn.MultiheadAttention(n_embed, n_head, batch_first=True)
        self.ln2 = RMSNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x):
        # Simple Global Attention for the Vision Encoder
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class JommarnVisionEncoder(nn.Module):
    """
    Mini-Vision Encoder trained from scratch.
    Target Parameters: ~3M
    """
    def __init__(self, img_size=224, patch_size=16, n_head=6, n_embed=192, n_layers=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, n_embed)
        
        # Jommarn-style spatial embeddings for images
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches, n_embed))
        
        # Depth layers for vision processing
        self.layers = nn.ModuleList([
            VisionBlock(n_head, n_embed) for _ in range(n_layers)
        ])
        
        self.ln_post = RMSNorm(n_embed)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        for layer in self.layers:
            x = layer(x)
            
        return self.ln_post(x) # (B, n_patches, n_embed)

if __name__ == '__main__':
    # Test initialization
    encoder = JommarnVisionEncoder()
    dummy_img = torch.randn(1, 3, 224, 224)
    tokens = encoder(dummy_img)
    print(f"Vision Tokens Shape: {tokens.shape}") # Expected: (1, 196, 192)
    print(f"Encoder Parameters: {sum(p.numel() for p in encoder.parameters()):,}")