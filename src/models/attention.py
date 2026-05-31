import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Partial Rotary Positional Embedding (p-RoPE) to a tensor.
    Rotates only the first half of the dimensions in each head.
    """
    B, T, H, D = x.shape
    d_rope = D // 2 # Partial RoPE: rotate only half
    x_rope = x[..., :d_rope]
    x_pass = x[..., d_rope:]

    # RoPE rotation: [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]
    x_rope_left = x_rope[..., 0::2]
    x_rope_right = x_rope[..., 1::2]
    
    # [B, T, H, d_rope//2] * [1, T, 1, d_rope//2]
    cos = cos[:T, :d_rope//2].unsqueeze(0).unsqueeze(2)
    sin = sin[:T, :d_rope//2].unsqueeze(0).unsqueeze(2)
    
    rotated_left = x_rope_left * cos - x_rope_right * sin
    rotated_right = x_rope_left * sin + x_rope_right * cos
    
    # Recombine
    x_rope = torch.stack([rotated_left, rotated_right], dim=-1).flatten(-2)
    return torch.cat([x_rope, x_pass], dim=-1)

class MultiHeadAttention(nn.Module):
    """
    Jommarn-Tiny Multi-Head Attention.
    
    Implements:
    1. Partial RoPE for enhanced spatial awareness.
    2. Hybrid Attention (Sliding Window or Global).
    3. Optimized vectorized computation (no more looping over heads).
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int) -> None:
        super().__init__()
        assert n_embed % n_head == 0
        self.n_head = n_head
        self.head_size = n_embed // n_head
        self.context_length = context_length
        self.window_size = context_length // 2 # Default sliding window size

        # Key, Query, Value projections in one go
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=False)
        
        # Causal mask buffer
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x: torch.Tensor, is_local: bool = False, rope_cos: torch.Tensor = None, rope_sin: torch.Tensor = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # QKV Projection
        qkv = self.qkv_proj(x) # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1)
        
        # Reshape for multi-head: (B, T, H, D)
        q = q.view(B, T, self.n_head, self.head_size)
        k = k.view(B, T, self.n_head, self.head_size)
        v = v.view(B, T, self.n_head, self.head_size)
        
        # Apply p-RoPE
        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)
            
        # Transpose for attention: (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled Dot-Product Attention
        # (B, H, T, D) @ (B, H, D, T) -> (B, H, T, T)
        attn_weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
        
        # Masking logic
        mask = self.tril[:T, :T]
        if is_local:
            # Sliding Window: only look back 'window_size' steps
            local_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=-self.window_size)
            mask = mask * local_mask
            
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)
        out = attn_weights @ v
        
        # Reshape back: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

if __name__ == '__main__':
    # Example Usage
    B, T, C = 2, 8, 32
    H = 4
    mha = MultiHeadAttention(n_head=H, n_embed=C, context_length=T)
    x = torch.randn(B, T, C)
    
    # Mock RoPE
    cos = torch.cos(torch.randn(T, (C//H)//4))
    sin = torch.sin(torch.randn(T, (C//H)//4))
    
    out = mha(x, is_local=True, rope_cos=cos, rope_sin=sin)
    print("MHA Output Shape:", out.shape)