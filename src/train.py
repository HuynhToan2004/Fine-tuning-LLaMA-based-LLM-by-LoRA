import torch
from transformers import TrainingArguments, Trainer
from .model_loader import load_model_and_tokenizer
from .data_utils import load_and_prepare_dataset, get_data_collator
from .config import TRAINING_ARGS

def train_lora_model():

    model, tokenizer = load_model_and_tokenizer()
    data = load_and_prepare_dataset(tokenizer, dataset_name='/data/npl/ICEK/VACNIC/src/data/assest/chatbot_instruction_prompts')
    training_args = TrainingArguments(
        per_device_train_batch_size = TRAINING_ARGS["per_device_train_batch_size"],
        gradient_accumulation_steps = TRAINING_ARGS["gradient_accumulation_steps"],
        num_train_epochs = TRAINING_ARGS["num_train_epochs"],
        learning_rate = TRAINING_ARGS["learning_rate"],
        fp16 = TRAINING_ARGS["fp16"],
        save_total_limit = TRAINING_ARGS["save_total_limit"],
        logging_steps = TRAINING_ARGS["logging_steps"],
        output_dir = TRAINING_ARGS["output_dir"],
        optim = TRAINING_ARGS["optim"],
        lr_scheduler_type = TRAINING_ARGS["lr_scheduler_type"],
        warmup_ratio = TRAINING_ARGS["warmup_ratio"]
    )

    trainer = Trainer(
        model = model,
        train_dataset = data,
        args = training_args,
        data_collator = get_data_collator(tokenizer)
    )

    model.config.use_cache = False
    trainer.train()
    trainer.save_model("experiments/lora_model")
    print("Finished training LoRA model.")
