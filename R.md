# สรุปสถาปัตยกรรม "Jommarn-Omni 206M" (Multimodal Evolution)

จากการปรับปรุงล่าสุด **Jommarn-Tiny** ได้วิวัฒนาการสู่ **Jommarn-Omni** ซึ่งเป็นโมเดลแบบ **Native Multimodal** ที่ทรงพลังที่สุดในขนาดกระทัดรัด โดยมีพารามิเตอร์รวมอยู่ที่ **206 ล้านพารามิเตอร์** ออกแบบมาเพื่อประมวลผลทั้ง **ข้อความ (Text)** และ **รูปภาพ (Vision)** โดยเน้นภาษาไทยเป็นพิเศษ

## 1. องค์ประกอบใหม่: Jommarn-Vision Encoder
เราได้เพิ่ม **Vision Encoder ที่สร้างขึ้นเองจากศูนย์ (Train from Scratch)** เพื่อรักษาความเบาและประสิทธิภาพ:
*   **Patch Embedding:** หั่นรูปภาพขนาด 224x224 ให้เป็น 196 tokens (Vision Tokens) โดยใช้เทคนิค Linear Projection เพื่อความเสถียรบน Hardware ทุกระดับ
*   **Intelligence Density:** ใช้ 3 เลเยอร์พิเศษที่มี RMSNorm และ SwiGLU เพื่อให้ข้อมูลภาพมีความหนาแน่นทางปัญญาสูงก่อนส่งต่อให้ตัว Thinker

## 2. ภาษาไทยระดับเทพ: Gemma-4 Tokenizer
เราได้อัปเกรด "พจนานุกรม" ของโมเดลให้เป็นระดับโลก:
*   **Gemma-4 Powered:** ใช้ Tokenizer จากโมเดล Gemma-4 ของ Google ซึ่งตัดคำภาษาไทยได้คมชัดและมีประสิทธิภาพสูงสุด
*   **Vocab Size:** 262,144 คำ (แก้ไขให้ตรงตามพจนานุกรมจริง เพื่อความเสถียรของ CUDA)

## 3. การทำงานแบบ Native Multimodal (Omni Architecture)
โมเดลประมวลผลภาพและข้อความใน "สมองเดียว":
*   **Hybrid Attention Schedule:** สลับเลเยอร์แบบ `Local (512 tokens) -> Global (1024 tokens)` เพื่อให้โมเดลจดจำรายละเอียดใกล้เคียงและเชื่อมโยงภาพรวมได้พร้อมกัน
*   **Weight Tying:** แชร์น้ำหนักระหว่าง Embedding และ Output Head ช่วยประหยัดพื้นที่และคงความฉลาดเท่าเดิม
*   **Next-Token Prediction:** ระบบการสอนที่ถูกต้อง (Target Shifting) บังคับให้โมเดลเรียนรู้การทำนายอนาคตจากบริบทจริง

## 4. ข้อมูลเชิงเทคนิค (Technical Specifications)
*   **Total Parameters:** ~206 ล้านพารามิเตอร์
*   **N_EMBED (มิติการเรียนรู้):** 512
*   **N_BLOCKS (ความลึก):** 14 เลเยอร์
*   **Context Length:** 1,024 Tokens
*   **Training Mode:** รองรับ Multi-GPU และ L40S Optimization (Batch 8 + Grad Accum 4)

---
*วิวัฒนาการโดย Gemini CLI - Jommarn-Omni Engine*

## คู่มือการรัน Jommarn-Omni (ฉบับสมบูรณ์)

### 1. การเตรียมสภาพแวดล้อม
```python
!pip install -q huggingface_hub transformers torchvision pillow tqdm h5py datasets
```

### 2. การอัปเดตโค้ดและล้างสถานะ (สำคัญเมื่อมีอัปเดต)
ทุกครั้งที่มีการแก้ไขสถาปัตยกรรม หรือเปลี่ยนเครื่องรัน ให้ทำตามนี้:
1.  **Restart Session** ของ Notebook/Studio
2.  รันคำสั่ง Pull:
```bash
%cd /teamspace/studios/this_studio/llm-training
!git pull origin main
```

### 3. การตั้งค่าและเริ่มการเทรน (ของจริง)
รันเซลล์นี้เพื่อเริ่มการเรียนรู้แบบ Multimodal:
```python
import os
from huggingface_hub import login

# 1. Login (ใช้ Token แบบ Write)
login("YOUR_HUGGINGFACE_TOKEN")

# 2. ตั้งชื่อ Repo สำหรับสำรองข้อมูลอัตโนมัติทุก 100 Step
os.environ["HF_REPO_ID"] = "Phonsiri/Jommarn-AI"

# 3. เริ่มการเทรน
!export PYTHONPATH=$PYTHONPATH:. && python scripts/train_transformer.py
```

### 4. การทดสอบสายตา (Inference)
เมื่อเทรนผ่านไปอย่างน้อย 100 steps สามารถทดสอบได้ทันที:
```bash
!export PYTHONPATH=$PYTHONPATH:. && python scripts/test_omni.py \
    --model "models/jommarn_omni_206m_l40s_latest.pt" \
    --image "your_image.jpg" \
    --prompt "รูปภาพนี้คือ"
```

---
**ข้อแนะนำสำหรับ L40S:** ไฟล์โมเดลมีขนาดประมาณ **916MB** และระบบ Auto-Push จะทำงานทุก 100 ก้าว โปรดตรวจสอบว่าอินเทอร์เน็ตเปิดอยู่เสมอ! 😈🔥📸