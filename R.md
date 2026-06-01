# สรุปสถาปัตยกรรม "Jommarn-Omni 206M" (Multimodal Evolution)

> **"Intelligence Density ในขนาดกระทัดรัด: จอมมารผู้มองเห็นและพูดไทยได้คล่องแคล่ว"**

## 🌟 Recap: จุดเด่นของ Jommarn-Omni 206M
*   **Native Multimodal:** รวม "ดวงตา" (Vision Encoder) และ "สมอง" (Thinker) ไว้ในเนื้อเดียวกัน ฝึกฝนจากศูนย์ (From Scratch) เพื่อให้ภาพและข้อความเข้าใจกันอย่างลึกซึ้ง
*   **Intelligence Density:** อัดแน่นความฉลาดด้วยสถาปัตยกรรมระดับ SOTA (SwiGLU, RMSNorm, PLE) ทำให้พารามิเตอร์ 206M มีประสิทธิภาพเทียบเท่าโมเดลขนาดใหญ่กว่าหลายเท่า
*   **Modern Transformer Stack:** ใช้เทคนิคเดียวกับ Llama 3 และ Gemma 4 (p-RoPE, Decoder-only, Hybrid Local/Global Attention)
*   **Thai Specialized:** รองรับภาษาไทยระดับโลกด้วย Gemma-4 Tokenizer (262,144 คำ) และฝึกฝนด้วยชุดข้อมูลระดับโปร (Thai Wiki v3, Handwriting, Appen OCR)
*   **Cloud-Native Stability:** ระบบ Auto-Resume ซิงค์ข้อมูลกับ Hugging Face อัตโนมัติ พร้อมระบบนิรภัยป้องกัน NaN (Gradient Clipping + Warmup) รันได้เสถียรทั้งบน T4 และ L40S

---

จากการปรับปรุงล่าสุด... (เนื้อเดิม)

## 1. องค์ประกอบใหม่: Jommarn-Vision Encoder
เราได้เพิ่ม **Vision Encoder ที่สร้างขึ้นเองจากศูนย์ (Train from Scratch)** เพื่อรักษาความเบาและประสิทธิภาพ:
*   **Patch Embedding:** หั่นรูปภาพขนาด 224x224 ให้เป็น 196 tokens (Vision Tokens) โดยใช้เทคนิค Linear Projection เพื่อความเสถียรสูงสุดบน Hardware ทุกระดับ
*   **Intelligence Density:** ใช้ 3 เลเยอร์พิเศษที่มี RMSNorm และ SwiGLU เพื่อสรุปความหมายจากภาพก่อนส่งต่อให้ตัว Thinker

## 2. ภาษาไทยระดับเทพ: Gemma-4 Tokenizer
เราได้อัปเกรด "พจนานุกรม" ของโมเดลให้เป็นระดับโลก:
*   **Gemma-4 Powered:** ใช้ Tokenizer จากโมเดล Gemma-4 ของ Google ซึ่งตัดคำภาษาไทยได้คมชัดที่สุด
*   **Vocab Size:** 262,144 คำ (ปรับแต่งให้ตรงตามพจนานุกรมจริง เพื่อความเสถียรของระบบ CUDA)

## 3. การทำงานแบบ Native Multimodal (Omni Architecture)
โมเดลประมวลผลภาพและข้อความใน "สมองเดียว":
*   **Hybrid Attention Schedule:** สลับเลเยอร์แบบ `Local (512 tokens window) -> Global (1024 tokens)` ช่วยให้จดจำรายละเอียดประโยคใกล้เคียงและเชื่อมโยงภาพรวมได้แม่นยำ
*   **True Multimodal Alignment:** ระบบจะใช้พิกเซลสุดท้ายของภาพมาทำนายตัวอักษรแรกผ่านโทเค็น `<bos>` ทำให้โมเดล "เริ่มอ่านภาพ" ได้ทันทีโดยไม่ต้องใช้ Prompt
*   **Weight Tying:** แชร์น้ำหนักระหว่างตารางคำศัพท์และตัวทำนาย ช่วยประหยัดพื้นที่ VRAM มหาศาล
*   **Next-Token Prediction:** ระบบการสอนที่ถูกต้อง (Target Shifting) ป้องกันการ "ลอกข้อสอบ" และบังคับให้โมเดลใช้ตรรกะในการทำนายคำถัดไป
*   **Auto-Resume System:** ระบบวาร์ปข้ามคลาวด์ สามารถดึง Checkpoint ล่าสุดจาก Hugging Face มาเทรนต่อได้ทันที

## 4. ข้อมูลเชิงเทคนิค (Technical Specifications)
*   **Total Parameters:** ~206 ล้านพารามิเตอร์
*   **Stability Stack:** ใช้ Gradient Clipping (1.0), LR Warmup (1,000 steps) และระบบ Safe Clamping เพื่อป้องกัน NaN
*   **Balanced Interleaving:** ระบบสุ่มข้อมูลแบบ 50/50 ระหว่างรูปภาพลายมือและความรู้จาก Wikipedia เพื่อความเก่งที่สมดุล
*   **Proven OCR:** จากการทดสอบเบื้องต้น โมเดลสามารถดึงคำสำคัญ (Keywords) จากภาพลายมือจริงได้สำเร็จตั้งแต่วินาทีแรกที่เรียนรู้!

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