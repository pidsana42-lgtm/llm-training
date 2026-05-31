# --- Jommarn-Omni 231M (Gemma-4 Powered) Configuration ---

# เป้าหมายพารามิเตอร์: ~231 Million
# จุดเด่น: ใช้ Tokenizer ของ Gemma-4 เพื่อการรองรับภาษาไทยระดับเทพ

# ตัวเลข VOCAB_SIZE ของ Gemma ปกติคือ 256,000 
# เราจะตั้งค่าเผื่อให้หารด้วย 64 ลงตัวเพื่อประสิทธิภาพ GPU (256000 + padding)
VOCAB_SIZE = 262144         # ตัวเลขจริงจาก tokenizer.json ของ Gemma-4 (2^18)
CONTEXT_LENGTH = 1024       
N_EMBED = 512               
N_HEAD = 8                  
N_BLOCKS = 14               

# Paths
TRAIN_PATH = "data/train/pile_train.h5"
DEV_PATH = "data/val/pile_dev.h5"
TOKENIZER_PATH = "tokenizer.json" # ไฟล์ที่ดึงมาจาก Gemma-4

# Training parameters (Optimized for L40S 48GB - God Mode)
T_BATCH_SIZE = 128          # ขยับขึ้นเป็น 128 เพื่อความเร็วสูงสุดบน L40S
T_CONTEXT_LENGTH = 1024     # ขยายหน้าต่างการเรียนรู้เป็น 1024 เต็มพิกัด
T_TRAIN_STEPS = 50000       # 50,000 ก้าวที่ทรงพลัง
T_EVAL_STEPS = 500         
T_EVAL_ITERS = 100         
T_LR_DECAY_STEP = 20000    
T_LR = 4e-4                 
T_LR_DECAYED = 4e-5        
T_OUT_PATH = "models/jommarn_omni_206m_l40s.pt"

# Device
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

default_config = {
    'vocab_size': VOCAB_SIZE,
    'context_length': CONTEXT_LENGTH,
    'n_embed': N_EMBED,
    'n_head': N_HEAD,
    'n_blocks': N_BLOCKS,
    'train_path': TRAIN_PATH,
    'dev_path': DEV_PATH,
    'tokenizer_path': TOKENIZER_PATH,
    't_batch_size': T_BATCH_SIZE,
    't_context_length': T_CONTEXT_LENGTH,
    't_train_steps': T_TRAIN_STEPS,
    't_eval_steps': T_EVAL_STEPS,
    't_eval_iters': T_EVAL_ITERS,
    't_lr_decay_step': T_LR_DECAY_STEP,
    't_lr': T_LR,
    't_lr_decayed': T_LR_DECAY_STEP,
    't_out_path': T_OUT_PATH,
    'device': DEVICE,
}