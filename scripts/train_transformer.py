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
losses = []
AVG_WINDOW = 64

# --- Multimodal Training Loop ---

# Using our new Master Loader that combines Wiki, Handwriting, and OCR
train_loader = get_master_loader(batch_size=config['t_batch_size'])

# Create the output directory if it does not exist.
os.makedirs(os.path.dirname(config['t_out_path']), exist_ok=True)

print(f"Starting Training on {config['device']}...")

pbar = tqdm(range(config['t_train_steps']))
train_iter = iter(train_loader)

for step in pbar:
    try:
        # Fetch a multimodal batch: (Images, Input_IDs, Targets)
        try:
            images, xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, xb, yb = next(train_iter)

        # Move to device
        images = images.to(config['device'])
        xb = xb.to(config['device'])
        yb = yb.to(config['device'])

        # Perform a forward pass with vision and text
        # If images are dummy (zeros from Wiki), the model handles it
        logits, loss = model(xb, images=images, targets=yb)

        # Record the loss for tracking.
        losses.append(loss.item())
        pbar.set_description(f"Loss: {np.mean(losses[-AVG_WINDOW:]):.4f}")

        # Backpropagate
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

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
                # Save a temporary checkpoint for pushing
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
hf_repo = os.getenv("HF_REPO_ID")
if hf_repo:
    print(f"\nAttempting to push model to Hugging Face: {hf_repo}")
    try:
        from scripts.push_to_hf import push_to_hub
        push_to_hub(repo_id=hf_repo, model_path=config['t_out_path'])
    except Exception as e:
        print(f"Failed to push to HF: {e}")