from datasets import load_dataset
from googletrans import Translator
from transformers import DataCollatorForLanguageModeling
from .config import DEVICE

def generate_prompt(data_point, translator):
    vn_prompt = translator.translate(data_point['prompt'], src='en', dest='vi').text
    vn_response = translator.translate(data_point['response'], src='en', dest='vi').text
    return f"""
        <|im_start|>system
        Bạn là một trợ lý AI hữu ích. Hãy trả lời người dùng một cách chính xác và chi tiết.
        <|im_end|>

        <|im_start|>user
        {vn_prompt}
        <|im_end|>

        <|im_start|>assistant
        {vn_response}
    """.strip()

def generate_and_tokenize_prompt(data_point, tokenizer, translator):
    full_prompt = generate_prompt(data_point, translator)
    tokenized_full_prompt = tokenizer(
        full_prompt,
        padding=True,
        truncation=True
    )
    return tokenized_full_prompt

def load_and_prepare_dataset(tokenizer, dataset_name='/data/npl/ICEK/VACNIC/src/data/assest/chatbot_instruction_prompts'):

    data = load_dataset(dataset_name)
    data = data['train'].shard(num_shards=50, index=0)
    data = data.filter(lambda sample: sample['response'] != '' and sample['prompt'] != '')
    data = data.shuffle()

    translator = Translator()
    data = data.map(lambda x: generate_and_tokenize_prompt(x, tokenizer, translator))
    
    return data

def get_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(tokenizer, mlm=False)
