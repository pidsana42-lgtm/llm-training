# สรุปสถาปัตยกรรม "Jommarn-Omni 203M" (Multimodal Evolution)

จากการปรับปรุงล่าสุด **Jommarn-Tiny** ได้วิวัฒนาการสู่ **Jommarn-Omni** ซึ่งเป็นโมเดลแบบ **Native Multimodal** ที่ทรงพลังที่สุดในขนาดกระทัดรัด โดยมีพารามิเตอร์รวมอยู่ที่ **203 ล้านพารามิเตอร์** ออกแบบมาเพื่อประมวลผลทั้ง **ข้อความ (Text)** และ **รูปภาพ (Vision)** โดยเน้นภาษาไทยเป็นพิเศษ

## 1. องค์ประกอบใหม่: Jommarn-Vision Encoder
เราได้เพิ่ม **Vision Encoder ที่สร้างขึ้นเองจากศูนย์ (Train from Scratch)** เพื่อรักษาความเบาและประสิทธิภาพ:
*   **Patch Embedding:** หั่นรูปภาพขนาด 224x224 ให้เป็น 196 tokens (Vision Tokens)
*   **Intelligence Density:** ใช้ 3 เลเยอร์พิเศษที่มี RMSNorm และ SwiGLU เพื่อให้ข้อมูลภาพมีความหนาแน่นทางปัญญาสูงก่อนส่งต่อให้ตัว Thinker

## 2. ภาษาไทยระดับเทพ: Gemma-4 Tokenizer
เราได้อัปเกรด "พจนานุกรม" ของโมเดลให้เป็นระดับโลก:
*   **Gemma-4 Powered:** ใช้ Tokenizer จากโมเดล Gemma-4 ของ Google ซึ่งตัดคำภาษาไทยได้คมชัดและมีประสิทธิภาพสูงสุด
*   **Vocab Size:** 256,128 คำ (ช่วยลดปัญหา Token ภาษาไทยแตกกระจาย)

## 3. การทำงานแบบ Native Multimodal (Omni Architecture)
โมเดลประมวลผลภาพและข้อความใน "สมองเดียว":
*   **Hybrid Attention Schedule:** สลับเลเยอร์แบบ `Local (512 tokens) -> Global (1024 tokens)` เพื่อให้โมเดลจดจำรายละเอียดใกล้เคียงและเชื่อมโยงภาพรวมได้พร้อมกัน
*   **Weight Tying:** แชร์น้ำหนักระหว่าง Embedding และ Output Head ช่วยประหยัดพื้นที่ไปกว่า 131 ล้านพารามิเตอร์ แต่คงความฉลาดเท่าเดิม

## 4. ข้อมูลเชิงเทคนิค (Technical Specifications)
*   **Total Parameters:** ~203 ล้านพารามิเตอร์
*   **N_EMBED (มิติการเรียนรู้):** 512
*   **N_BLOCKS (ความลึก):** 14 เลเยอร์
*   **Context Length:** 1,024 Tokens
*   **Training Speed:** ปรับแต่งมาเพื่อรันบน **Kaggle GPU T4 x 2** ได้อย่างสมบูรณ์แบบ

---
*วิวัฒนาการโดย Gemini CLI - Jommarn-Omni Engine*

## คู่มือการรัน Jommarn-Omni บน Cloud (Kaggle/Colab)

### 1. การเตรียมสภาพแวดล้อม (Environment Setup)
```python
!pip install -q huggingface_hub transformers torchvision pillow tqdm h5py datasets
```

### 2. การดึง Tokenizer และ Dataset
ยืนยันตัวตนกับ Hugging Face เพื่อดึงพจนานุกรม Gemma-4:
```python
from huggingface_hub import login
login("YOUR_HUGGINGFACE_TOKEN")

# ดาวน์โหลด Tokenizer
!python scripts/download_tokenizer.py
```

### 3. การใช้งาน Master Data Loader (Wiki + Handwriting + OCR)
เราได้เตรียมระบบรวมข้อมูลจาก 3 แหล่งสำคัญไว้ในไฟล์เดียว:
*   **Thai Wiki v3:** ฐานความรู้ภาษาไทย
*   **Thai Handwriting:** ระบบอ่านลายมือไทย
*   **Appen Thai Document OCR:** ระบบอ่านเอกสารราชการและธุรกิจ

เรียกใช้งานในสคริปต์เทรนของคุณ:
```python
from scripts.master_data_loader import get_master_loader
train_loader = get_master_loader(batch_size=32)
```

### 4. เริ่มต้นการฝึกฝน (Training)
```python
!python scripts/train_transformer.py
```
*โมเดลจะถูกบันทึกไว้ใน `models/jommarn_omni_231m_thai.pt` (ขนาดไฟล์จริงประมาณ 800MB - 1GB)*

### 5. การทดสอบ (Inference)
ทดสอบให้จอมมารอ่านลายมือหรือเอกสาร:
```python
!python scripts/generate_text.py --model_path models/jommarn_omni_231m_thai.pt --input_text "รูปภาพนี้คือเอกสารที่เขียนว่า"
```

---
**ข้อแนะนำพิเศษ:** สำหรับ Kaggle อย่าลืมเปิดปุ่ม **Accelerator: GPU T4 x2** และตั้งค่า **Internet: On** ในเมนูขวามือครับ!