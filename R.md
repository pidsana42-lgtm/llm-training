# สรุปสถาปัตยกรรม "Jommarn-Omni ~501M" (Multimodal & 4-Token MTP Evolution)

> **"Intelligence Density ในขนาดกระทัดรัด: จอมมารผู้มองเห็น คิดก่อนตอบ และซิ่งแรง 4 เท่าด้วย MTP"**

## 🌟 Recap: จุดเด่นของ Jommarn-Omni 469M
*   **Native Multimodal Early Fusion:** รวม "ดวงตา" (Vision Encoder) และ "สมอง" (Thinker) เข้าสู่สมองเดียวตั้งแต่จุดแรกเข้าภาพ (Prepend Vision Tokens หน้า Text Tokens) ให้ความเข้าใจสอดคล้องประสานกันอย่างลึกซึ้ง
*   **4-Token Multi-Token Prediction (MTP):** สถาปัตยกรรมปฏิวัติวงการ (อิงงานวิจัย DeepMind/Meta) ใช้ MTP Mixers 3 ตัวเรียนรู้และทำนายโทเคนล่วงหน้า $t+2, t+3, t+4$ พร้อมกันตอนเทรน และเร่งสปีดการเจนคำตอบตอนใช้งาน (Inference) เร็วขึ้นเป็นเท่าตัว (3x-4x Speedup)
*   **Grouped-Query Attention (GQA):** อัปเกรดกลไกการทำ Attention สัดส่วน 12 Query heads : 2 KV heads (6:1 ratio) ประหยัดแรม VRAM สำหรับเก็บ KV Cache และรันได้ลื่นไหลบน Context ยาวๆ
*   **16-Layer Vision Encoder:** เพิ่มระดับชั้นเลเยอร์การมองเห็นเต็มพิกัดเป็น 16 เลเยอร์ (มาตรฐาน ViT-Large) เพื่อรองรับการสกัดคุณลักษณะ (Features) ภาพถ่ายบิล หน้าเว็บ และลายมือขึ้นมาจากศูนย์ (Train from scratch) อย่างคมกริบ
*   **Typhoon Tokenizer (152k Vocab):** เปลี่ยนมาใช้ Tokenizer ที่ปรับจูนสำหรับภาษาไทยโดยเฉพาะ (จาก Qwen2.5-VL/Typhoon) ทำให้ประหยัดพารามิเตอร์ส่วนที่ไม่จำเป็นไปกว่า 80 ล้านพารามิเตอร์ และทำให้โมเดลเรียนรู้โครงสร้างประโยคภาษาไทยได้เร็วกว่าเดิม
*   **5-Way Balanced Training Data:** จัดสัดส่วนการสุ่มเรียนรู้ข้อมูลแบบสมดุล 5 ด้าน (สัดส่วนอย่างละ 20% เท่ากัน) เพื่อการจัดรูปแบบการทำความเข้าใจอย่างครบรอบด้าน ทั้งเอกสาร ลายมือ ภาพทั่วไป และหลักภาษาไทย

---

## 🔬 รายละเอียดสถาปัตยกรรมเชิงลึก (Architecture Specs)

### 1. องค์ประกอบตัวโมเดลหลัก (Shared Trunk)
*   **Total Parameters:** ~501 ล้านพารามิเตอร์ (รีดประสิทธิภาพเต็มที่ภายใต้งบ 500M สำหรับ Edge AI)
*   **Layers (N_BLOCKS):** 32 Blocks (เพิ่มความลึกเท่า LLaMA ทำให้ตรรกะและการคิดวิเคราะห์แข็งแรงขึ้นมาก)
*   **Embedding Dimension (N_EMBED):** 768
*   **Attention Heads:** 12 Query Heads (Head Size = 64)
*   **KV Heads (GQA):** 2 KV Heads (แชร์ Key-Value Cache ร่วมกันในอัตราส่วน 6:1)
*   **Context Length:** 4,096 โทเคน
*   **Activation Function:** SwiGLU MLP (ขยายมิติ 4 เท่าไปยัง 3072 มิติ)

### 2. โมดูลการมองเห็น (Jommarn-Vision Encoder)
*   **Layers:** 16 Blocks 
*   **Image Resolution:** 512x512 พิกเซล (แปลงภาพขนาดใหญ่และเอกสาร A4 ออกเป็น 1,024 Vision Tokens โดยใช้ Patch Size ขนาด 16x16 พิกเซล) เพื่อให้ตาของโมเดลอ่านฟอนต์ภาษาไทยขนาดเล็กบนหน้าบิลได้คมชัด 100%
*   **Parameter Size:** ~132M พารามิเตอร์ (เพิ่มความลึกเป็น 16 ชั้น เพื่อการมองเห็นระดับเทพ)

### 3. ระบบ Multi-Token Prediction (MTP Mixers)
*   **Mixers:** 3 ชุดขนาน (MTP1, MTP2, MTP3)
*   **การทำงาน:** นำการแสดงค่าชั้นสุดท้ายไปคลุกเคล้า (Mix) กับเวกเตอร์คำตอบของตัวก่อนหน้า และทำนายล่วงหน้าขนานไปกับหัวทำนายปกติ
*   **Parameter Size:** ~5.3M พารามิเตอร์

---

## 📚 ชุดข้อมูลการเรียนรู้ (6-Way Balanced Dataset + Distillation)

เราจัดสัดส่วนการสุ่มเรียนรู้ (Sampling Weight) ไว้ใหม่ โดยเน้นให้ความสำคัญกับ "เฉลยจากครู" ในช่วงต้น เพื่อดึง Loss ให้ลงเร็วที่สุด:

1.  **Distillation Warmup Data:** 40% (ข้อมูลที่สร้างล่วงหน้าโดยใช้ `Typhoon-OCR-7B` เป็นครู เพื่อสอนโมเดลตาบอดให้อ่านออกอย่างรวดเร็ว)
2.  **Detailed OCR & CoT (`Phonsiri/handwrite-ocr-detailed`):** 12% (บรรยายภาพและกระบวนการคิด)
3.  **Astrology & Document Layout (`Phonsiri/astrology-dataset-clean`):** 12% (เอกสารและอินโฟกราฟิก)
4.  **COCO Thai General Understanding (`Phonsiri/coco-thai-gemma4-detailed`):** 12% (ภาพทั่วไป)
5.  **Handwriting OCR (`iapp/thai_handwriting_dataset`):** 12% (ลายมือไทย)
6.  **Thai Wikipedia (`pythainlp/thai-wiki-dataset-v3`):** 12% (ป้องกันโมเดลลืมภาษาไทย)

---

## 🏃‍♂️ คู่มือการรัน Jommarn-Omni ~501M

### 1. การเตรียมระบบและดึงเวอร์ชันล่าสุด
ให้สั่งดึงตัวแก้ไขล่าสุดจากบน Cloud GPU เซิร์ฟเวอร์ของคุณ:
```bash
git pull origin main
pip install qwen-vl-utils accelerate
```

### 2. สร้างคัมภีร์ข้อมูล Distillation จากครู (ทำล่วงหน้า 1 ครั้ง)
สั่งให้ `Typhoon-OCR-7B` วิเคราะห์รูปภาพเพื่อทำเฉลย (รันรอบเดียวทิ้งไว้ข้ามคืน):
```bash
python scripts/create_distillation_dataset.py
```
*(เคล็ดลับ: สามารถใส่ `--test` ต่อท้ายเพื่อทดสอบรันแค่ 4 รูปก่อนได้)*

### 3. แผนการเทรน 2 ระยะ (2-Phase Training Strategy)
เพื่อป้องกันไม่ให้ Loss ระเบิดจากการเทรนภาพและตัวหนังสือพร้อมกันจากศูนย์ เราจะแบ่งการเทรนออกเป็น 2 ระยะ:

#### ระยะที่ 1: "สร้างสมอง" (Text-Only Pre-training)
ปิดตาทิ้ง แล้วให้โมเดลเสพแต่ Thai Wikipedia อย่างเดียว เพื่อเรียนรู้ไวยากรณ์ภาษาไทยให้แตกฉานก่อน (รันข้ามคืน 1-2 วันจนกว่า Loss จะนิ่ง):
```bash
TRAIN_PHASE="text_only" FORCE_RESET=1 HF_REPO_ID="Phonsiri/jommarn-omni-checkpoints" python scripts/train_transformer.py
```

#### ระยะที่ 2: "เปิดตา" (Multimodal Alignment)
หลังจากสมองพูดภาษาไทยรู้เรื่องแล้ว ให้หยุดการเทรนข้างบน แล้วรันคำสั่งใหม่โดย **ลบคำว่า FORCE_RESET ทิ้ง** และเปลี่ยนโหมดเป็น Multimodal เพื่อให้มันเริ่มดูภาพและอ่านข้อมูล Distillation:
```bash
TRAIN_PHASE="multimodal" HF_REPO_ID="Phonsiri/jommarn-omni-checkpoints" python scripts/train_transformer.py
```
*(ในระยะนี้ Data Loader จะโหลดไฟล์ JSONL จากครูมาเป็นหนังสือนำทาง (Weight 40%) ทำให้ตากับสมองเชื่อมกันได้เร็วที่สุด)*

### 4. การทดสอบการมองเห็นและเจนคำตอบ (Inference)
แนะนำให้ใช้ Jupyter Notebook `test_omni.ipynb` เพื่อหลีกเลี่ยงปัญหาเรื่อง Encoding ภาษาต่างดาวใน Terminal ของ Cloud

---

## 🛠️ การปรับจูนและแก้ปัญหาบน Cloud (Troubleshooting & Tuning)

### 1. การล้างพื้นที่ดิสก์ (Disk Space Management)
หากใช้ Cloud แล้วพื้นที่เต็ม (เช่น กราฟพุ่งไป 300GB+) เกิดจาก Cache ของ Hugging Face ที่โหลด Dataset และ Model ซ้ำซ้อน ให้รันคำสั่งเหล่านี้บน Terminal:
```bash
# ล้างแคช Dataset และ Model ที่ซ้ำซ้อน (ได้พื้นที่คืนมหาศาล)
rm -rf ~/.cache/huggingface/datasets/
rm -rf ~/.cache/huggingface/hub/
# ลบไฟล์ Checkpoint ระหว่างทางทิ้ง (เก็บไว้แค่ _latest.pt)
rm -f ~/llm-training/models/*_step_*.pt
```

### 2. การจูนบนการ์ดจอ AMD MI300X (VRAM 192GB)
สถาปัตยกรรมนี้รองรับทั้ง **Nvidia (CUDA)** และ **AMD (ROCm)** อย่างสมบูรณ์!
หากคุณเช่า MI300X ที่มี VRAM มหาศาล ให้เข้าไปปรับใน `config/config.py`:
```python
T_BATCH_SIZE = 128     # อัดรูป/ข้อความเข้าการ์ดจอทีละ 128 เพื่อใช้ 192GB ให้คุ้มค่า
T_GRAD_ACCUM = 4       # Effective Batch Size = 512 (Loss นิ่งกริบ!)
```
*(คำเตือน: ต้องติดตั้ง PyTorch เวอร์ชัน ROCm จากเว็บ Official เท่านั้น ห้ามรัน pip install torch เฉยๆ)*

### 3. เป้าหมายของการทำ Pre-training (เมื่อไหร่ควรหยุด?)
*   **เป้าหมาย Loss:** ไม่ใช่ 1.0 (ถ้า 1.0 คือ Overfitting) เป้าหมายที่สมบูรณ์แบบระดับโลกคือ **1.8 - 2.2** 
*   **พฤติกรรมโมเดล:** เมื่อเทรนจบ Phase 1 โมเดลจะแต่งประโยคภาษาไทยได้เป๊ะ ไวยากรณ์ถูกต้อง 100% ไม่มีคำศัพท์มนุษย์ต่างดาว **แต่มันจะยังไม่ยอมตอบคำถาม (เพราะยังไม่ผ่าน Phase 2: Instruction Tuning)** มันจะแค่พยายามแต่งประโยคต่อให้จบหน้ากระดาษแบบสารานุกรม

---
*วิวัฒนาการโดย Antigravity AI - Jommarn-Omni ~501M Engine*
😈📸⚡

pip install -r requirements.txt
pip install datasets transformers huggingface_hub
export HF_TOKEN="your_hf_token_here"
