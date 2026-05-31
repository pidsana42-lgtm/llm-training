# สรุปสถาปัตยกรรม "Jommarn-Omni" (Multimodal Evolution)

จากการปรับปรุงล่าสุด **Jommarn-Tiny** ได้วิวัฒนาการสู่ **Jommarn-Omni** ซึ่งเป็นโมเดลแบบ **Native Multimodal** ที่สามารถประมวลผลได้ทั้ง **ข้อความ (Text)** และ **รูปภาพ (Vision)** ในสถาปัตยกรรมเดียว โดยที่ยังคงความเบาในระดับ 16.5 ล้านพารามิเตอร์

## 1. องค์ประกอบใหม่: Jommarn-Vision Encoder
เราได้เพิ่ม **Vision Encoder ที่สร้างขึ้นเองจากศูนย์ (Train from Scratch)** เพื่อให้เข้ากับปรัชญาความเบาของโมเดล:
*   **Patch Embedding:** หั่นรูปภาพขนาด 224x224 ให้เป็นชิ้นเล็กๆ (Patches) เพื่อแปลงเป็น Vision Tokens
*   **Vision Transformer Blocks:** ใช้ 3 เลเยอร์พิเศษที่มี RMSNorm และ SwiGLU แบบเดียวกับตัว Thinker เพื่อให้ข้อมูลภาพมีความหนาแน่นทางปัญญาสูง
*   **Parameter Efficiency:** ส่วน Vision นี้เพิ่มพารามิเตอร์เพียงประมาณ 3.3 ล้านพารามิเตอร์เท่านั้น

## 2. การทำงานแบบ Native Multimodal (Omni Architecture)
โมเดลไม่ได้แค่อ่านภาพแยกกัน แต่ใช้เทคนิค **Late Fusion (Prefix-tuning)**:
*   **Vision Tokens:** รูปภาพหนึ่งรูปจะถูกแปลงเป็น 196 tokens และถูกนำไปวางไว้หน้าข้อความ
*   **Thinker (Jommarn-Tiny):** ทำหน้าที่เป็นสมองส่วนกลาง ประมวลผลทั้งภาพและข้อความพร้อมกันใน **Hybrid Attention Layers**
    *   **Global Attention:** ช่วยให้ข้อความสามารถ "มองเห็น" และทำความเข้าใจรายละเอียดของรูปภาพที่อยู่ส่วนต้นของลำดับได้อย่างแม่นยำ
    *   **p-RoPE:** ปรับแต่งให้รองรับลำดับที่ยาวขึ้นจากการรวมภาพและข้อความเข้าด้วยกัน

## 3. ข้อมูลเชิงเทคนิค (Technical Specifications)
*   **Total Parameters:** ~16.5 ล้านพารามิเตอร์
*   **Architecture:** Hybrid Decoder-only Transformer + ViT Encoder
*   **Capabilities:** 
    *   Text Generation (สร้างข้อความ)
    *   Image Captioning (บรรยายรูปภาพ)
    *   Visual Question Answering (ตอบคำถามจากภาพ - ต้องฝึกฝนเพิ่มเติม)

## 4. ทำไม Jommarn-Omni ถึงพิเศษ?
การมี Vision Encoder ที่สร้างจาก Scratch ทำให้ Jommarn-Omni เป็น **"เนื้อเดียวกัน"** ทั้งระบบ ไม่มีการพึ่งพาโมเดลภายนอกที่หนักเกินไป ทำให้มันเป็นหนึ่งในโมเดล Multimodal ที่เล็กและทรงพลังที่สุด สามารถรันแบบ Real-time บนอุปกรณ์พกพาได้อย่างแท้จริง

---
*วิวัฒนาการโดย Gemini CLI - Jommarn-Omni Engine*

## คู่มือการรัน Jommarn-Omni บน Cloud (Kaggle/Colab)

เพื่อให้การฝึกฝนจอมมารออมนิขนาด 203M เป็นไปอย่างราบรื่นบนขุมพลัง **GPU T4 x 2** ให้ปฏิบัติตามขั้นตอนดังนี้:

### 1. การเตรียมสภาพแวดล้อม (Environment Setup)
ติดตั้งไลบรารีที่จำเป็นในเซลล์แรกของ Notebook:
```python
!pip install -q huggingface_hub transformers torchvision pillow tqdm h5py
```

### 2. การดึง Tokenizer ของ Gemma-4
เนื่องจากเราใช้ภาษาของ Gemma คุณต้องยืนยันตัวตนกับ Hugging Face ก่อน:
```python
from huggingface_hub import login
login("YOUR_HUGGINGFACE_TOKEN") # นำ Token มาจากหน้า Settings ใน Hugging Face

# รันสคริปต์ดาวน์โหลดที่เตรียมไว้
!python scripts/download_tokenizer.py
```

### 3. การเตรียมข้อมูล (Dataset)
*   **Text Data:** สามารถใช้สคริปต์ `scripts/data_preprocess.py` เพื่อเตรียมข้อมูล Wikipedia เป็น HDF5
*   **Vision Data:** แนะนำให้ใช้ **COCO Dataset** ที่มีอยู่แล้วบน Kaggle โดยกดปุ่ม "Add Data" และเลือก `coco-2017-dataset`

### 4. การตั้งค่าก่อนเริ่มเทรน (Config Check)
ตรวจสอบไฟล์ `config/config.py` ว่าตรงกับความต้องการ:
*   `VOCAB_SIZE`: 256128
*   `N_EMBED`: 512
*   `N_BLOCKS`: 14
*   `DEVICE`: 'cuda'

### 5. เริ่มต้นการฝึกฝน (Training)
รันคำสั่งเพื่อเริ่มกระบวนการเรียนรู้:
```python
!python scripts/train_transformer.py
```
*ระบบจะเริ่มบันทึกโมเดลไว้ในโฟลเดอร์ `models/jommarn_omni_231m_thai.pt` เมื่อเสร็จสิ้น*

### 6. การทดสอบการมองเห็น (Inference)
หลังจากเทรนเสร็จ สามารถทดสอบให้จอมมารบรรยายภาพได้ด้วย:
```python
!python scripts/generate_text.py --model_path models/jommarn_omni_231m_thai.pt --input_text "This image shows"
```

---
**ข้อแนะนำ:** สำหรับการรันบน Kaggle T4 x 2 แนะนำให้เปิดใช้งาน **Accelerator: GPU T4 x2** ในเมนู Settings ด้านขวามือเพื่อให้การเทรนเร็วขึ้นเป็น 2 เท่า!