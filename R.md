# สรุปสถาปัตยกรรม "Jommarn-Omni 206M" (Multimodal Evolution)

จากการปรับปรุงล่าสุด **Jommarn-Tiny** ได้วิวัฒนาการสู่ **Jommarn-Omni** ซึ่งเป็นโมเดลแบบ **Native Multimodal** ที่ทรงพลังที่สุดในขนาดกระทัดรัด โดยมีพารามิเตอร์รวมอยู่ที่ **206 ล้านพารามิเตอร์** ออกแบบมาเพื่อประมวลผลทั้ง **ข้อความ (Text)** และ **รูปภาพ (Vision)** โดยเน้นภาษาไทยเป็นพิเศษ

## 1. องค์ประกอบใหม่: Jommarn-Vision Encoder
เราได้เพิ่ม **Vision Encoder ที่สร้างขึ้นเองจากศูนย์ (Train from Scratch)** เพื่อรักษาความเบาและประสิทธิภาพ:
*   **Patch Embedding:** หั่นรูปภาพขนาด 224x224 ให้เป็น 196 tokens (Vision Tokens) โดยใช้เทคนิค Linear Projection เพื่อความเสถียรสูงสุด
*   **Intelligence Density:** ใช้ 3 เลเยอร์พิเศษที่มี RMSNorm และ SwiGLU เพื่อสรุปความหมายจากภาพก่อนส่งต่อให้ตัว Thinker

## 2. ภาษาไทยระดับเทพ: Gemma-4 Tokenizer
เราได้อัปเกรด "พจนานุกรม" ของโมเดลให้เป็นระดับโลก:
*   **Gemma-4 Powered:** ใช้ Tokenizer จากโมเดล Gemma-4 ของ Google ซึ่งตัดคำภาษาไทยได้คมชัดที่สุด
*   **Vocab Size:** 262,144 คำ (ปรับแต่งให้ตรงตามพจนานุกรมจริง เพื่อความเสถียรของระบบ CUDA)

## 3. การทำงานแบบ Native Multimodal (Omni Architecture)
โมเดลประมวลผลภาพและข้อความใน "สมองเดียว":
*   **Hybrid Attention Schedule:** สลับเลเยอร์แบบ `Local (512 tokens window) -> Global (1024 tokens)` ช่วยให้จดจำรายละเอียดประโยคใกล้เคียงและเชื่อมโยงภาพรวมได้แม่นยำ
*   **Weight Tying:** แชร์น้ำหนักระหว่างตารางคำศัพท์และตัวทำนาย ช่วยประหยัดพื้นที่ VRAM มหาศาล
*   **Next-Token Prediction:** ระบบการสอนที่ถูกต้อง (Target Shifting) ป้องกันการ "ลอกคำตอบ" และบังคับให้โมเดลใช้ตรรกะในการทำนายคำถัดไป
*   **Auto-Resume System:** ระบบวาร์ปข้ามคลาวด์ สามารถดึง Checkpoint ล่าสุดจาก Hugging Face มาเทรนต่อได้ทันทีเมื่อเปลี่ยนเครื่อง

## 4. ข้อมูลเชิงเทคนิค (Technical Specifications)
*   **Total Parameters:** ~206 ล้านพารามิเตอร์
*   **N_EMBED (มิติการเรียนรู้):** 512
*   **N_BLOCKS (ความลึก):** 14 เลเยอร์
*   **Context Length:** 1,024 Tokens
*   **Mixed Data:** ฝึกด้วย Thai Wiki v3 (ความรู้) สลับกับ Thai Handwriting (การอ่านลายมือ)

---
*วิวัฒนาการโดย Gemini CLI - Jommarn-Omni Engine*

## คู่มือการรัน Jommarn-Omni (ฉบับสมบูรณ์)

### 1. การเตรียมสภาพแวดล้อม
```python
!pip install -q huggingface_hub transformers torchvision pillow tqdm h5py datasets
```

### 2. การอัปเดตโค้ดและล้างสถานะ (บังคับทำเมื่อมีอัปเดต)
1.  **Restart Session** ของ Notebook เพื่อล้างแรม GPU
2.  รันคำสั่งดึงตัวแก้ไขล่าสุด:
```bash
%cd /teamspace/studios/this_studio/llm-training
!git pull origin main
```

### 3. การตั้งค่าและเริ่มการเทรน (รองรับการ Resume อัตโนมัติ)
```python
import os
from huggingface_hub import login

# 1. Login (ใช้ Token แบบ Write)
login("YOUR_HUGGINGFACE_TOKEN")

# 2. ตั้งชื่อ Repo (ระบบจะดึงไฟล์จากที่นี่มาเทรนต่อหากในเครื่องไม่มี)
os.environ["HF_REPO_ID"] = "Phonsiri/Jommarn-AI"

# 3. เริ่มการเทรนระดับเทพ (L40S Optimization)
!export PYTHONPATH=$PYTHONPATH:. && python scripts/train_transformer.py
```

### 4. การทดสอบสายตา (Inference)
ทดสอบความฉลาดของจอมมารหลังจากผ่าน 100 ก้าวแรก:
```bash
!export PYTHONPATH=$PYTHONPATH:. && python scripts/test_omni.py \
    --model "models/jommarn_omni_206m_l40s_latest.pt" \
    --image "test.jpeg" \
    --prompt "รูปภาพนี้คือ"
```

---
**ข้อแนะนำ:** จอมมารออมนิจะทำการสำรองความรู้ขึ้น Hugging Face ทุกๆ **100 Step** โปรดเปิดเน็ตไว้ให้จอมมารด้วยนะครับ! 😈🔥📸

เทสส
!export PYTHONPATH=$PYTHONPATH:/teamspace/studios/this_studio/llm-training && python /teamspace/studios/this_studio/llm-training/scripts/test_omni.py --model "/teamspace/studios/this_studio/llm-training/models/jommarn_omni_206m_l40s_latest.pt" --image "/teamspace/studios/this_studio/llm-training/config/test.jpeg" --prompt "ประเทศไทย"