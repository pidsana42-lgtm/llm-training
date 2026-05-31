import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import numpy as np
from config.config import default_config as config
from src.models.transformer import JommarnOmni as Transformer
from scripts.master_data_loader import get_master_loader
from typing import Dict

# --- Initialize the Model and Print Parameters ---

model = Transformer(
    n_head=config['n_head'],
    n_embed=config['n_embed'],
    context_length=config['context_length'],
    vocab_size=config['vocab_size'],
    N_BLOCKS=config['n_blocks']
)

# Multi-GPU Support: Re-enabled for L40S Power
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for maximum acceleration!")
    model = torch.nn.DataParallel(model)

model = model.to(config['device'])

# Print the total number of parameters
inner_model = model.module if hasattr(model, 'module') else model
total_params = sum(p.numel() for p in inner_model.parameters())
print(f"Total number of parameters in the model: {total_params:,}")

# --- Optimizer Setup and Loss Tracking ---

optimizer = torch.optim.AdamW(model.parameters(), lr=config['t_lr'])
# scaler = torch.amp.GradScaler('cuda') # Optional: For Mixed Precision if needed
losses = []
AVG_WINDOW = 64

# --- Multimodal Training Loop ---

# Using our new Master Loader
train_loader = get_master_loader(batch_size=config['t_batch_size'])

# Create the output directory if it does not exist.
os.makedirs(os.path.dirname(config['t_out_path']), exist_ok=True)

print(f"Starting Training on {config['device']} with physical batch size {config['t_batch_size']}...")
grad_accum_steps = config.get('t_grad_accum', 1)
print(f"Gradient Accumulation enabled: {grad_accum_steps} steps (Effective batch size: {config['t_batch_size'] * grad_accum_steps})")

pbar = tqdm(range(config['t_train_steps']))
train_iter = iter(train_loader)
hf_repo = os.getenv("HF_REPO_ID")

for step in pbar:
    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0
    
    try:
        # Gradient Accumulation Loop
        for _ in range(grad_accum_steps):
            try:
                images, tokens, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, tokens, _ = next(train_iter)

            # Target Shifting: xb is input (0 to T-1), yb is target (1 to T)
            # This is the core of language modeling!
            xb = tokens[:, :-1].to(config['device'])
            yb = tokens[:, 1:].to(config['device'])
            images = images.to(config['device'])

            # Forward pass
            with torch.amp.autocast('cuda'):
                logits, loss = model(xb, images=images, targets=yb)
            
            # Scale loss for accumulation
            loss = loss / grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

        # Update weights after accumulation
        optimizer.step()
        
        # Record the loss for tracking.
        losses.append(accum_loss)
        pbar.set_description(f"Loss: {np.mean(losses[-AVG_WINDOW:]):.4f}")

        # Decay the learning rate
        if step == config['t_lr_decay_step']:
            print('\nDecaying learning rate')
            for g in optimizer.param_groups:
                g['lr'] = config['t_lr_decayed']
                
        # Save checkpoint periodically
        if step > 0 and step % config['t_eval_steps'] == 0:
            checkpoint_path = config['t_out_path'].replace(".pt", f"_step_{step}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            
        # Periodic Push to Hugging Face every 100 steps
        if step > 0 and step % 100 == 0 and hf_repo:
            print(f"\nStep {step}: Periodic push to Hugging Face...")
            try:
                temp_checkpoint = config['t_out_path'].replace(".pt", "_latest.pt")
                torch.save(model.state_dict(), temp_checkpoint)
                from scripts.push_to_hf import push_to_hub
                push_to_hub(repo_id=hf_repo, model_path=temp_checkpoint)
            except Exception as e:
                print(f"Failed periodic push: {e}")

    except Exception as e:
        print(f"Error at step {step}: {e}")
        continue

# --- Save Final Model ---

torch.save(
    {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses,
        'steps': len(losses),
    },
    config['t_out_path']
)
print(f"Saved final model to {config['t_out_path']}")

# --- Automatic Push to Hugging Face (Optional) ---
if hf_repo:
    print(f"\nAttempting to push model to Hugging Face: {hf_repo}")
    try:
        from scripts.push_to_hf import push_to_hub
        push_to_hub(repo_id=hf_repo, model_path=config['t_out_path'])
    except Exception as e:
        print(f"Failed to push to HF: {e}")