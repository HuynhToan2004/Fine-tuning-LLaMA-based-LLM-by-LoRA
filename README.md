# Fine-tuning-LLaMA-based-LLM-by-LoRA

A demonstration of **LoRA** (Low-Rank Adaptation) fine-tuning on a LLaMA-based large language model (7B parameters) combined with **4-bit quantization**. This approach allows training large models on limited GPU resources while preserving decent model performance.

---

## Overview

This project shows how to:

1. **Load** a pre-trained LLaMA-like model in **4-bit** mode (using [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)).
2. **Fine-tune** the model with **LoRA** ([PEFT library](https://github.com/huggingface/peft)) to update only a small set of low-rank parameters.
3. **Translate** prompts and responses on the fly (English → Vietnamese) with [googletrans](https://pypi.org/project/googletrans/).
4. **Train** using the [Hugging Face Transformers Trainer](https://github.com/huggingface/transformers) on a small subset of a dataset.
5. **Generate** text responses via the fine-tuned model.

By leveraging **4-bit quantization**, the GPU memory footprint is drastically reduced, enabling experimentation with 7B+ parameter models using a single or small number of GPUs.

---

## Features

- **4-bit quantization** via [BitsAndBytesConfig](https://huggingface.co/docs/transformers/main_classes/quantization) to reduce VRAM usage.
- **LoRA** for parameter-efficient fine-tuning, updating only rank-\(r\) matrices.
- **Prompt engineering** in “assistant-user” chat style.
- On-the-fly **translation** of dataset from English to Vietnamese, helpful for multilingual or localized tasks.
- Fully **modular code** structure:
  - `config.py` for all hyperparameters and model configurations.
  - `model_loader.py` for loading/quantizing the model and adding LoRA.
  - `data_utils.py` for data processing, translation, and tokenization.
  - `train.py` for the training logic using `Trainer`.
  - `inference.py` for generating text.
  - `scripts/` contains runnable scripts to train or do inference from the command line.

---

## Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/yourusername/Fine-tuning-LLaMA-based-LLM-by-LoRA.git
   cd Fine-tuning-LLaMA-based-LLM-by-LoRA
   ```

2. **Install** dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

1. **Training** :
   ```bash
   cd scripts
   python run_training.py
   ```
2. **Inference** :
   ```bash
   cd scripts
   python run_inference.py
   ```



