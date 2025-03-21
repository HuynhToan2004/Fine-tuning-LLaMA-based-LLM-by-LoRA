
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, prepare_model_for_kbit_training
from .config import MODEL_NAME, bnb_config, lora_config, DEVICE

def load_model_and_tokenizer():

    # Load model 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="cuda",   
        trust_remote_code=True,
        quantization_config=bnb_config
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, lora_config)

    gen_config = model.generation_config
    gen_config.max_new_tokens = 200
    gen_config.temperature = 0.7
    gen_config.top_p = 0.7
    gen_config.num_return_sequences = 1
    gen_config.pad_token_id = tokenizer.eos_token_id
    gen_config.eos_token_id = tokenizer.eos_token_id
    
    model.to(DEVICE)

    return model, tokenizer
