import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.transformer_block import Block, RMSNorm
from src.models.vision_encoder import JommarnVisionEncoder

def precompute_rope_freqs(head_size: int, context_length: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes RoPE frequencies for the rotary positional embeddings.
    """
    d_rope = head_size // 2
    freqs = 1.0 / (theta ** (torch.arange(0, d_rope, 2).float() / d_rope))
    t = torch.arange(context_length + 256) # Increased context for Vision tokens
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

class JommarnOmni(nn.Module):
    """
    Jommarn-Omni: A Native Multimodal (Vision-Text) Model.
    
    Combines Jommarn-Tiny (Thinker) with Jommarn-Vision (Encoder).
    Total Parameters: ~16.5M
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int, vocab_size: int, N_BLOCKS: int) -> None:
        super().__init__()
        self.context_length = context_length
        self.N_BLOCKS = N_BLOCKS
        self.n_head = n_head
        self.head_size = n_embed // n_head
        
        # 1. Vision Encoder (Scratch)
        self.vision_encoder = JommarnVisionEncoder(n_head=n_head, n_embed=n_embed)
        
        # 2. Text Thinker
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.attn_blocks = nn.ModuleList([
            Block(n_head, n_embed, context_length + 256, layer_id=i) 
            for i in range(N_BLOCKS)
        ])
        
        self.final_norm = RMSNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        
        # RoPE
        cos, sin = precompute_rope_freqs(self.head_size, context_length)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)
        
        # Weight tying
        self.token_embed.weight = self.lm_head.weight

    def forward(self, idx: torch.Tensor, images: torch.Tensor = None, targets: torch.Tensor = None):
        B, T = idx.shape
        
        # Safety Fix: Clamp tokens to ensure they are within [0, vocab_size-1]
        # This prevents "index out of bounds" errors if the tokenizer/dataset has unexpected tokens.
        idx = torch.clamp(idx, 0, self.token_embed.num_embeddings - 1)
        
        # Text Embeddings
        x = self.token_embed(idx)
        
        # Vision Integration (Late Fusion / Prefix)
        if images is not None:
            v_tokens = self.vision_encoder(images) # (B, n_patches, n_embed)
            # Prepend vision tokens to text tokens
            x = torch.cat([v_tokens, x], dim=1)
            
        # Global RoPE context
        curr_T = x.shape[1]
        cos, sin = self.rope_cos[:curr_T], self.rope_sin[:curr_T]
        
        # Hybrid Attention Processing
        for i, block in enumerate(self.attn_blocks):
            is_local = (i % 2 == 0) and (i < self.N_BLOCKS - 1)
            # Vision tokens always get Global focus indirectly through block orchestration
            x = block(x, is_local=is_local, rope_cos=cos, rope_sin=sin)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        # For targets, we care about predicting from the last vision token onwards
        if images is not None:
            n_patches = v_tokens.shape[1]
            # Take logits starting from the last vision token to predict the first text token
            logits = logits[:, n_patches-1 : -1, :] 
            
        loss = None
        if targets is not None:
            # Shift targets to align with logits
            # If multimodal, logits already shifted relative to text in forward pass
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T).long())
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, images=None, max_new_tokens=100):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond, images=images)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

if __name__ == '__main__':
    # Test Multimodal Forward
    model = JommarnOmni(n_head=6, n_embed=192, context_length=128, vocab_size=50304, N_BLOCKS=6)
    idx = torch.randint(0, 50304, (1, 10))
    img = torch.randn(1, 3, 224, 224)
    logits, _ = model(idx, images=img)
    print(f"Logits shape with Vision: {logits.shape}")
    print(f"Jommarn-Omni Total Parameters: {sum(p.numel() for p in model.parameters()):,}")