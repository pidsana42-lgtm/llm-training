# Jommarn-Omni 203M: Multimodal Thai Intelligence 😈🇹🇭🖼️

Jommarn-Omni is a state-of-the-art **Native Multimodal (Vision + Text)** model designed for high "Intelligence Density" in a compact 203M parameter footprint. It is specifically optimized for Thai language understanding, handwriting OCR, and document analysis.

## 🚀 Key Evolutionary Features

Developed from a base Transformer, Jommarn-Omni integrates modern architectural breakthroughs used in models like Gemma 4 and Llama 3:

*   **Native Multimodal:** A built-in **Jommarn-Vision Encoder** that allows the model to "see" and "think" about images and text in a single semantic space.
*   **SwiGLU Activation:** Replaces standard ReLU for more expressive and efficient learning.
*   **RMSNorm:** Provides superior training stability and speed compared to traditional LayerNorm.
*   **Hybrid Attention Schedule:** Interleaved **Sliding Window (Local)** and **Global Attention** layers to balance detail-oriented processing with long-range context (1,024 tokens).
*   **Partial RoPE (p-RoPE):** Advanced rotary positional embeddings for precise spatial awareness.
*   **Gemma-4 Tokenizer:** Utilizes the massive 256k vocabulary from Google's Gemma-4, ensuring flawless Thai language support without token fragmentation.

## 📊 Model Specifications

| Parameter | Value |
|-----------|-------|
| **Total Parameters** | ~203 Million |
| **Architecture** | Decoder-only Transformer + ViT Encoder |
| **Embedding Dim (N_EMBED)** | 512 |
| **Layers (N_BLOCKS)** | 14 |
| **Attention Heads** | 8 |
| **Context Length** | 1,024 Tokens |
| **Vocabulary Size** | 256,128 (Gemma-4) |

## 📚 Specialized Thai Datasets

Jommarn-Omni is designed to be trained on a powerful combination of Thai data:
1.  **Thai Wiki v3:** For deep linguistic foundations and general knowledge.
2.  **Thai Handwriting Dataset:** For mastering human-written Thai OCR.
3.  **Appen Thai Document OCR:** For professional-grade official document understanding.

## 🛠️ Usage & Cloud Training (Kaggle/Colab)

Jommarn-Omni is perfectly sized for **GPU T4 x 2** (32GB VRAM) environments.

### 1. Environment Setup
```bash
pip install -q huggingface_hub transformers torchvision pillow tqdm h5py datasets
```

### 2. Get the Gemma-4 Tokenizer
```python
from huggingface_hub import login
login("YOUR_HF_TOKEN")

# Run our automation script
!python scripts/download_tokenizer.py
```

### 3. Master Data Pipeline
Load all Thai datasets seamlessly:
```python
from scripts.master_data_loader import get_master_loader
train_loader = get_master_loader(batch_size=32)
```

### 4. Start Training
```bash
python scripts/train_transformer.py
```

## 📜 Documentation
For a detailed technical summary and Thai language guide, please refer to [R.md](R.md).

---
*Developed and Refactored by Gemini CLI - Jommarn-Omni Engine*