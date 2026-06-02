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
    t = torch.arange(context_length + 1280) # Increased context for 512x512 Vision tokens (1024 patches) + safety buffer
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

class JommarnOmni(nn.Module):
    """
    Jommarn-Omni: A Native Multimodal (Vision-Text) Model.
    
    Combines Jommarn-Tiny (Thinker) with Jommarn-Vision (Encoder).
    Total Parameters: ~16.5M (original) / ~469M (configured)
    """
    def __init__(self, n_head: int, n_embed: int, context_length: int, vocab_size: int, N_BLOCKS: int, n_kv_head: int = 2, img_size: int = 512, v_layers: int = 8) -> None:
        super().__init__()
        self.context_length = context_length
        self.N_BLOCKS = N_BLOCKS
        self.n_head = n_head
        self.head_size = n_embed // n_head
        
        # 1. Vision Encoder (Scratch)
        self.vision_encoder = JommarnVisionEncoder(img_size=img_size, n_head=n_head, n_embed=n_embed, n_layers=v_layers)
        
        # 2. Text Thinker
        self.token_embed = nn.Embedding(vocab_size, n_embed)
        self.attn_blocks = nn.ModuleList([
            Block(n_head, n_embed, context_length + 1280, layer_id=i, n_kv_head=n_kv_head) 
            for i in range(N_BLOCKS)
        ])
        
        self.final_norm = RMSNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        
        # ✅ 3. Multi-Token Prediction (MTP) Stack for n=4
        # We need 3 mixers: MTP1 (predicts t+2), MTP2 (predicts t+3), MTP3 (predicts t+4)
        self.mtp_mixers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * n_embed, n_embed, bias=False),
                RMSNorm(n_embed),
                nn.GELU(),
                nn.Linear(n_embed, n_embed, bias=False)
            ) for _ in range(3)
        ])
        
        # RoPE
        cos, sin = precompute_rope_freqs(self.head_size, context_length)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)
        
        # Weight tying
        self.token_embed.weight = self.lm_head.weight
        
        # Cache for fast inference MTP steps
        self._last_h = None

    def forward(self, idx: torch.Tensor, images: torch.Tensor = None, targets: torch.Tensor = None, v_tokens: torch.Tensor = None):
        B, T = idx.shape
        
        # Safety Fix: Clamp tokens to ensure they are within [0, vocab_size-1]
        idx = torch.clamp(idx, 0, self.token_embed.num_embeddings - 1)
        
        # Text Embeddings
        x = self.token_embed(idx)
        
        # Vision Integration (Late Fusion / Prefix)
        if v_tokens is None and images is not None:
            v_tokens = self.vision_encoder(images)
            
        if v_tokens is not None:
            # Prepend vision tokens to text tokens
            x = torch.cat([v_tokens, x], dim=1)
            
        # Global RoPE context
        curr_T = x.shape[1]
        cos, sin = self.rope_cos[:curr_T], self.rope_sin[:curr_T]
        
        # Hybrid Attention Processing
        for i, block in enumerate(self.attn_blocks):
            is_local = (i % 2 == 0) and (i < self.N_BLOCKS - 1)
            x = block(x, is_local=is_local, rope_cos=cos, rope_sin=sin)
            
        h = self.final_norm(x)
        
        # Extract text representations (skip vision tokens if present)
        if v_tokens is not None:
            n_patches = v_tokens.shape[1]
            h_text = h[:, n_patches:, :]
        else:
            h_text = h
            
        # Store the last hidden state for fast MTP inference
        self._last_h = h_text[:, -1, :]
            
        logits = self.lm_head(h_text)
        
        loss = None
        if targets is not None:
            # Clamp targets
            targets = torch.clamp(targets, 0, self.lm_head.out_features - 1)
            
            # For 4-Token MTP, we need targets for:
            # y1 (t+1), y2 (t+2), y3 (t+3), y4 (t+4)
            # This requires sequence length to be at least 4.
            if targets.shape[1] >= 4:
                # Slicing targets:
                y1 = targets[:, 0:-3]
                y2 = targets[:, 1:-2]
                y3 = targets[:, 2:-1]
                y4 = targets[:, 3:]
                
                # Slicing standard logits to align with y1
                h_text_sliced = h_text[:, :-3, :]
                logits_1 = logits[:, :-3, :]
                
                loss_1 = F.cross_entropy(logits_1.reshape(-1, logits_1.size(-1)), y1.reshape(-1).long())
                
                # Recursive Gated MTP predictions
                current_h = h_text_sliced
                mtp_losses = []
                targets_list = [y2, y3, y4]
                
                # Loop through the 3 MTP mixers
                for k in range(3):
                    # Condition on the previous target token (e.g. y1 for MTP1, y2 for MTP2, etc.)
                    cond_token = y1 if k == 0 else targets_list[k-1]
                    emb_cond = self.token_embed(cond_token)
                    
                    # Mix previous layer representation with target token embedding
                    current_h = self.mtp_mixers[k](torch.cat([current_h, emb_cond], dim=-1))
                    
                    # Project to vocabulary
                    logits_k = self.lm_head(current_h)
                    y_k = targets_list[k]
                    
                    loss_k = F.cross_entropy(logits_k.reshape(-1, logits_k.size(-1)), y_k.reshape(-1).long())
                    mtp_losses.append(loss_k)
                
                # Combined Loss: Primary loss + 0.3 * sum(MTP losses)
                loss = loss_1 + 0.3 * sum(mtp_losses)
                logits = logits_1 # Return primary logits for logging compat
            else:
                # Fallback to standard 1-token prediction if sequence is too short
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1).long())
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, images=None, max_new_tokens=100, temperature=0.8):
        self.eval()
        # Pre-calculate vision tokens once to save compute
        v_tokens = self.vision_encoder(images) if images is not None else None
        
        # MTP generates 4 tokens per loop step -> cut steps to a quarter!
        steps = max(1, max_new_tokens // 4)
        for _ in range(steps):
            idx_cond = idx[:, -self.context_length:]
            logits, _ = self(idx_cond, v_tokens=v_tokens) # Populates self._last_h
            
            # 1. Sample Token 1 (Standard Head)
            logits_curr = logits[:, -1, :] / temperature
            probs_curr = F.softmax(logits_curr, dim=-1)
            next_id_curr = torch.multinomial(probs_curr, num_samples=1)
            
            # Store generated tokens in a list
            gen_ids = [next_id_curr]
            h_state = self._last_h.unsqueeze(1) # (B, 1, n_embed)
            
            # 2. Predict Tokens 2, 3, and 4 recursively using the 3 MTP mixers
            for k in range(3):
                emb_prev = self.token_embed(gen_ids[-1]) # Embed the last predicted token
                h_state = self.mtp_mixers[k](torch.cat([h_state, emb_prev], dim=-1))
                
                logits_k = self.lm_head(h_state).squeeze(1) / temperature
                probs_k = F.softmax(logits_k, dim=-1)
                next_id_k = torch.multinomial(probs_k, num_samples=1)
                gen_ids.append(next_id_k)
                
            # Concatenate all 4 newly predicted tokens to sequence
            new_tokens = torch.cat(gen_ids, dim=1)
            idx = torch.cat((idx, new_tokens), dim=1)
            
            # Early stopping check if any generated token is EOS (id 1)
            if any(tid.item() == 1 for tid in gen_ids):
                break
        return idx

if __name__ == '__main__':
    # Test Multimodal Forward (GQA 485M Model Config)
    model = JommarnOmni(n_head=12, n_embed=768, context_length=128, vocab_size=262144, N_BLOCKS=22, n_kv_head=2)
    idx = torch.randint(0, 262144, (1, 10))
    img = torch.randn(1, 3, 512, 512)
    logits, _ = model(idx, images=img)
    print(f"Logits shape with Vision (GQA): {logits.shape}")
    print(f"Jommarn-Omni Total Parameters: {sum(p.numel() for p in model.parameters()):,}")