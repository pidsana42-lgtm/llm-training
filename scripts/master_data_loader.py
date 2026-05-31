import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from transformers import PreTrainedTokenizerFast
import glob
import json

class JommarnMasterDataset(Dataset):
    """
    Unified Dataset for Jommarn-Omni.
    Handles:
    1. Thai Wiki (Text only)
    2. Thai Handwriting (Vision + Text)
    3. Appen Thai Document OCR (Vision + Text)
    """
    def __init__(self, tokenizer_path="tokenizer.json", img_size=224, mode="multimodal"):
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        
        # Gemma tokenizer usually doesn't have a default pad token set in the json
        # We set it to <|pad|> or <|endoftext|> to avoid TypeError during batching
        if "<|pad|>" in self.tokenizer.get_vocab():
            self.tokenizer.pad_token = "<|pad|>"
        else:
            self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
            
        self.img_size = img_size
        self.mode = mode # 'multimodal' or 'text_only'
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def tokenize(self, text, max_len=512):
        # Prepend <bos> (id 2 for Gemma) to help model know where to start
        return self.tokenizer(
            "<bos>" + text + "<|endoftext|>",
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt"
        )["input_ids"].squeeze(0)

class HandwritingSource(JommarnMasterDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Handwriting Dataset...")
        self.data = load_dataset("iapp/thai_handwriting_dataset", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = self.transform(item["image"].convert("RGB"))
        tokens = self.tokenize(item["text"])
        return img, tokens, tokens # img, input_ids, targets

class WikiSource(JommarnMasterDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Thai Wiki Dataset...")
        self.data = load_dataset("pythainlp/thai-wiki-dataset-v3", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Wiki has no image, we provide a dummy/zero tensor
        img = torch.zeros(3, self.img_size, self.img_size)
        tokens = self.tokenize(self.data[idx]["text"])
        return img, tokens, tokens

class AppenOCRSource(JommarnMasterDataset):
    def __init__(self, base_path="/kaggle/input/ocr-image-data-for-thai-documents", **kwargs):
        super().__init__(**kwargs)
        print(f"Checking Appen OCR at {base_path}...")
        # Note: Exact folder structure depends on Kaggle extraction
        self.img_files = glob.glob(os.path.join(base_path, "**/*.jpg"), recursive=True)
        # Mocking labels since access to specific JSON depends on user's Kaggle setup
        # In practice, you'd load the accompanying JSON/CSV for labels.

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        # Placeholder text: In training, replace with actual label from JSON
        tokens = self.tokenize("ตัวอย่างข้อความจากเอกสาร") 
        return img, tokens, tokens

def get_master_loader(batch_size=16):
    """
    Creates a combined loader that samples from all sources in a BALANCED way.
    Ensures that Vision data appears as frequently as Text data, despite dataset size differences.
    """
    ds_hw = HandwritingSource()
    ds_wiki = WikiSource()
    
    # Combined dataset
    master_ds = ConcatDataset([ds_hw, ds_wiki])
    
    # Calculate weights for balancing (Minority class gets higher weight)
    num_hw = len(ds_hw)
    num_wiki = len(ds_wiki)
    total = num_hw + num_wiki
    
    # Weights for each sample in the concatenated dataset
    # We want 50% chance for HW and 50% for Wiki
    weights = [total / num_hw] * num_hw + [total / num_wiki] * num_wiki
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=total, replacement=True)
    
    print(f"Balanced Sampler Active: Handwriting ({num_hw}) vs Wiki ({num_wiki})")
    print(f"Sampling Ratio: 1 Vision task for every 1 Text task (approx)")

    return DataLoader(
        master_ds, 
        batch_size=batch_size, 
        sampler=sampler, # Using the new balanced sampler
        num_workers=2,
        pin_memory=True
    )

if __name__ == "__main__":
    # Test
    try:
        loader = get_master_loader(batch_size=4)
        img, input_ids, targets = next(iter(loader))
        print(f"Batch Loaded Success!")
        print(f"Images: {img.shape}, Tokens: {input_ids.shape}")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This script is designed to run in an environment with access to the datasets.")