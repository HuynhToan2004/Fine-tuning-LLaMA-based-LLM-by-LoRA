import torch
from .model_loader import load_model_and_tokenizer
from .config import DEVICE

def generate_text(prompt: str, max_new_tokens=200):
    model, tokenizer = load_model_and_tokenizer()

    encoding = tokenizer(prompt, return_tensors='pt').to(DEVICE)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids = encoding.input_ids,
            attention_mask = encoding.attention_mask,
            max_new_tokens = max_new_tokens
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
