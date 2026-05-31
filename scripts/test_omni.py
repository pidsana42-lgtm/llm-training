import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import PreTrainedTokenizerFast
from src.models.transformer import JommarnOmni as Transformer
import argparse
import os

def test_omni(model_path, image_path, prompt, vocab_size=262144, n_embed=512, n_head=8, n_blocks=14):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Tokenizer
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")
    
    # 2. Load Model
    model = Transformer(
        n_head=n_head, 
        n_embed=n_embed, 
        context_length=1024, 
        vocab_size=vocab_size, 
        N_BLOCKS=n_blocks
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    # Handle DataParallel or raw state_dict
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # Clean state_dict if it was saved with DataParallel
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # 3. Process Image (Optional)
    image_tensor = None
    if image_path and os.path.exists(image_path):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device) # (1, 3, 224, 224)
        print(f"Vision Mode: Image '{image_path}' attached.")
    else:
        print("Text-Only Mode: No image attached.")

    # 4. Process Text
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

    # 5. Generate with Early Stopping
    print(f"Jommarn-Omni is thinking...")
    max_new_tokens = 50
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.encode("<|endoftext|>")[0]
    
    with torch.no_grad():
        generated_ids = input_ids
        for _ in range(max_new_tokens):
            # Pass to model
            logits, _ = model(generated_ids, images=image_tensor)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated_ids = torch.cat((generated_ids, next_id), dim=1)
            
            # Check for EOS (Early Stopping) - More robust check
            next_token_text = tokenizer.decode(next_id.item())
            if next_id.item() == eos_token_id or "<|endoftext|>" in next_token_text or "<unused" in next_token_text:
                print("Jommarn-Omni finished speaking.")
                break
        
    result = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    
    # Post-process cleanup
    result = result.replace("<|endoftext|>", "").strip()
    # Remove any trailing junk or repeated ' คือ'
    print(f"\n--- Result ---\n{result}\n--------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/jommarn_omni_206m_l40s_latest.pt")
    parser.add_argument("--image", type=str, default=None, help="Path to test image (optional)")
    parser.add_argument("--prompt", type=str, default="รูปภาพนี้คือ", help="Thai prompt")
    
    args = parser.parse_args()
    test_omni(args.model, args.image, args.prompt)