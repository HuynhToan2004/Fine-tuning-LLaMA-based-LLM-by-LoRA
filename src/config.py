import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig

MODEL_NAME = "/data/npl/ICEK/VACNIC/src/data/assest/vinallama-7b-chat"

# Cấu hình BitsAndBytes cho 4-bit 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Cấu hình LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "up_proj", "down_proj", "gate_proj"
    ],
    bias='none',
    task_type='CAUSAL_LM'
)

TRAINING_ARGS = {
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 1,
    "learning_rate": 2e-4,
    "fp16": True,
    "save_total_limit": 3,
    "logging_steps": 1,
    "output_dir": "experiments",
    "optim": "paged_adamw_8bit",
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
