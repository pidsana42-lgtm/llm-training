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
import random

class JommarnMasterDataset(Dataset):
    """
    Unified Dataset for Jommarn-Omni.
    Handles:
    1. Thai Wiki (Text only)
    2. Thai Handwriting (Vision + Text)
    3. Appen Thai Document OCR (Vision + Text)
    """
    def __init__(self, tokenizer_path="tokenizer.json", img_size=512, mode="multimodal"):
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
        
        # Wrap raw OCR text in the unified QA template
        prompt = "จงถอดความลายมือในรูปภาพนี้"
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {item['text']}"
        tokens = self.tokenize(text_format)
        
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

class DetailedOcrSource(JommarnMasterDataset):
    """
    Detailed OCR Source (Phonsiri/handwrite-ocr-detailed)
    Randomly select a task configuration to train on all relationships.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Phonsiri/handwrite-ocr-detailed Dataset...")
        self.data = load_dataset("Phonsiri/handwrite-ocr-detailed", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = self.transform(item["image"].convert("RGB"))
        
        # Randomly select a task to train all relationships
        task = random.choice(["ocr_only", "caption_th", "caption_en", "thinking_ocr"])
        
        if task == "ocr_only":
            prompt = "จงอ่านข้อความทั้งหมดที่ปรากฏอยู่ในภาพนี้"
            target = item["ocr_text"]
        elif task == "caption_th":
            prompt = "จงอธิบายรายละเอียดของภาพนี้อย่างละเอียด"
            target = item["detailed_caption_thai"]
        elif task == "caption_en":
            prompt = "Describe the details of this image in English."
            target = item["detailed_caption_en"]
        else: # thinking_ocr
            prompt = "จงวิเคราะห์ภาพและถอดข้อความภาษาไทยออกมาทีละขั้นตอน"
            target = (
                f"<think>\n{item.get('think_process', '')}\n</think>\n\n"
                f"**ข้อความที่ถอดได้:**\n{item.get('ocr_text', '')}\n\n"
                f"**คำบรรยายภาพ:**\n{item.get('detailed_caption_thai', '')}"
            )
            
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {target}"
        tokens = self.tokenize(text_format)
        return img, tokens, tokens

class AstrologyDatasetSource(JommarnMasterDataset):
    """
    Astrology & Document Layout Dataset (Phonsiri/astrology-dataset-clean)
    Provides OCR, Layout Captioning, and Category Classification.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Phonsiri/astrology-dataset-clean Dataset...")
        self.data = load_dataset("Phonsiri/astrology-dataset-clean", split="train")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = self.transform(item["image"].convert("RGB"))
        
        # Randomly select a task to train on different document dimensions
        task = random.choice(["ocr_text", "layout_caption", "classify"])
        
        if task == "ocr_text":
            prompt = "จงอ่านและถอดข้อความทั้งหมดจากเอกสารหรือรูปภาพนี้"
            target = item["text"]
        elif task == "layout_caption":
            prompt = "จงอธิบายโครงสร้างและรายละเอียดของรูปภาพหรือเอกสารนี้อย่างละเอียด"
            target = item["caption"]
        else: # classify
            prompt = "รูปภาพนี้จัดอยู่ในหมวดหมู่อะไร"
            target = f"หมวดหมู่: {item['category']}"
            
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {target}"
        tokens = self.tokenize(text_format)
        return img, tokens, tokens

class CocoThaiDetailedSource(JommarnMasterDataset):
    """
    COCO Thai Detailed Captions Dataset (Phonsiri/coco-thai-gemma4-detailed)
    Provides General Scene Understanding in Thai and English with dynamic fallback cleaning.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Loading Phonsiri/coco-thai-gemma4-detailed Dataset...")
        self.data = load_dataset("Phonsiri/coco-thai-gemma4-detailed", split="train")

    def __len__(self):
        return len(self.data)

    def _clean_text(self, text):
        if not text:
            return ""
        text_lower = text.lower()
        # Skip placeholders & instructions
        if "thai description" in text_lower or "english description" in text_lower:
            return ""
        if "combine all" in text_lower or "translate the" in text_lower:
            return ""
        return text.strip()

    def __getitem__(self, idx):
        item = self.data[idx]
        img = self.transform(item["image"].convert("RGB"))
        
        # Clean candidates
        th_detailed = self._clean_text(item.get("detailed_caption_thai"))
        en_detailed = self._clean_text(item.get("detailed_caption_en"))
        th_think = self._clean_text(item.get("think_process"))
        th_orig = self._clean_text(item.get("original_caption"))
        
        # Combine Thai and English detailed captions
        captions = []
        
        # Resolve best Thai caption
        th_caption = th_detailed if (th_detailed and len(th_detailed) > 10) else th_think
        if not th_caption or len(th_caption) < 10:
            th_caption = th_orig
            
        en_caption = en_detailed
        
        if th_caption:
            captions.append(th_caption)
            
        if en_caption and len(en_caption) > 10 and en_caption not in captions:
            # Check if one is a substring of the other to avoid duplicate/mixed sentences
            if th_caption and (th_caption in en_caption or en_caption in th_caption):
                # Use the longer one
                if len(en_caption) > len(th_caption):
                    captions = [en_caption]
                else:
                    captions = [th_caption]
            else:
                captions.append(en_caption)
                
        if not captions:
            # Absolute fallback
            captions = [th_orig if th_orig else "รายละเอียดรูปภาพ"]
            
        target = "\n\n".join(captions)
        prompt = "จงบรรยายรายละเอียดของภาพนี้อย่างละเอียด"
        
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {target}"
        tokens = self.tokenize(text_format, max_len=512)
        return img, tokens, tokens

class DistillationSource(JommarnMasterDataset):
    """
    Teacher Distillation Dataset (Warmup Data)
    Reads from the pre-generated JSONL file containing Typhoon-OCR-7B responses.
    """
    def __init__(self, data_path="data/distilled_warmup_data.jsonl", **kwargs):
        super().__init__(**kwargs)
        print(f"Loading Distillation Dataset from {data_path}...")
        self.items = []
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.items.append(json.loads(line))
        else:
            print(f"⚠️ Warning: Distillation file not found at {data_path}. Please run create_distillation_dataset.py first.")

    def __len__(self):
        return len(self.items) if self.items else 1 # Avoid division by zero

    def __getitem__(self, idx):
        if not self.items:
            # Fallback if file doesn't exist yet
            img = torch.zeros(3, self.img_size, self.img_size)
            tokens = self.tokenize("ผู้ใช้: จงวิเคราะห์ภาพนี้\n\nผู้ช่วย: ยังไม่มีข้อมูล Distillation")
            return img, tokens, tokens
            
        item = self.items[idx]
        try:
            img = self.transform(Image.open(item["image_path"]).convert("RGB"))
        except Exception:
            img = torch.zeros(3, self.img_size, self.img_size)
            
        prompt = "จงวิเคราะห์ภาพและถอดข้อความภาษาไทยออกมาทีละขั้นตอน"
        target = item["teacher_text"]
            
        text_format = f"ผู้ใช้: {prompt}\n\nผู้ช่วย: {target}"
        tokens = self.tokenize(text_format)
        return img, tokens, tokens

def get_master_loader(batch_size=16, phase="multimodal"):
    """
    Creates a combined loader.
    If phase == 'text_only', it only loads Thai Wikipedia to pre-train the language model.
    If phase == 'multimodal', it balances across all vision-language sources.
    """
    if phase == "text_only":
        print("🚀 PHASE 1: TEXT ONLY (Language Pre-training)")
        ds_wiki = WikiSource()
        return DataLoader(
            ds_wiki, 
            batch_size=batch_size, 
            shuffle=True, # Wiki can just be shuffled directly
            num_workers=2,
            pin_memory=True
        )
        
    # Phase 2: Multimodal Balanced Load
    print("🌌 PHASE 2: MULTIMODAL ALIGNMENT")
    ds_hw = HandwritingSource()
    ds_wiki = WikiSource()
    ds_detailed = DetailedOcrSource()
    ds_astrology = AstrologyDatasetSource()
    ds_coco = CocoThaiDetailedSource()
    ds_distill = DistillationSource()
    
    # Combined dataset
    master_ds = ConcatDataset([ds_hw, ds_wiki, ds_detailed, ds_astrology, ds_coco, ds_distill])
    
    num_hw = len(ds_hw)
    num_wiki = len(ds_wiki)
    num_detailed = len(ds_detailed)
    num_astrology = len(ds_astrology)
    num_coco = len(ds_coco)
    num_distill = len(ds_distill)
    total = num_hw + num_wiki + num_detailed + num_astrology + num_coco + num_distill
    
    # Weight Distribution (Distillation takes 40% of the batches to stabilize learning fast)
    w_hw = (total * 0.12) / max(1, num_hw)
    w_wiki = (total * 0.12) / max(1, num_wiki)
    w_detailed = (total * 0.12) / max(1, num_detailed)
    w_astrology = (total * 0.12) / max(1, num_astrology)
    w_coco = (total * 0.12) / max(1, num_coco)
    w_distill = (total * 0.40) / max(1, num_distill) # 40% for Warmup Distillation!
    
    weights = (
        [w_hw] * num_hw + 
        [w_wiki] * num_wiki + 
        [w_detailed] * num_detailed +
        [w_astrology] * num_astrology +
        [w_coco] * num_coco +
        [w_distill] * num_distill
    )
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=total, replacement=True)
    
    print(f"Balanced Sampler Active:")
    print(f" - Distillation (Teacher Data): {num_distill} rows (Sample weight: 40%)")
    print(f" - Detailed OCR: {num_detailed} rows (Sample weight: 12%)")
    print(f" - Astrology Layout: {num_astrology} rows (Sample weight: 12%)")
    print(f" - COCO General: {num_coco} rows (Sample weight: 12%)")
    print(f" - Handwriting: {num_hw} rows (Sample weight: 12%)")
    print(f" - Thai Wiki: {num_wiki} rows (Sample weight: 12%)")

    return DataLoader(
        master_ds, 
        batch_size=batch_size, 
        sampler=sampler,
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