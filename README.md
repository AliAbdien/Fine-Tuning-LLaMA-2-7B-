# 🚀 Fine-Tuning LLaMA 2 (7B)

This repository contains the code and configurations for fine-tuning **LLaMA 2 (7B)** using **LoRA (Low-Rank Adaptation)**. The model has been fine-tuned leveraging **Hugging Face's Transformers** library and **PEFT (Parameter Efficient Fine-Tuning)** techniques.

## 📂 Project Structure

```
📦 Fine-Tuning-LLaMA2-7B
│── 📜 adapter_config.json  # LoRA adapter configuration
│── 📜 special_tokens_map.json  # Special tokens configuration
│── 📜 tokenizer.json       # Tokenizer vocabulary and merges
│── 📜 tokenizer_config.json  # Tokenizer settings
│── 📜 training_args.bin    # Training arguments and hyperparameters
│── 📜 LLMA.ipynb           # Jupyter Notebook for fine-tuning and inference
```

## 📌 Features

✅ Fine-tunes **LLaMA 2 (7B)** using **LoRA**  
✅ Efficiently trains with reduced GPU memory requirements  
✅ Supports **custom tokenizers and special tokens**  
✅ Implements **text generation and evaluation**  
✅ Includes **checkpoints** for continued training  

## 🛠 Installation

Make sure you have Python 3.8+ installed. Then, install the required dependencies:

```bash
pip install torch transformers peft accelerate safetensors
```

## 🚀 Fine-Tuning LLaMA 2 (7B)

You can fine-tune the model using the provided Jupyter Notebook:

```bash
jupyter notebook LLMA.ipynb
```

Alternatively, you can run the training script in Python:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load the base model and tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load LoRA adapter
adapter_model = PeftModel.from_pretrained(model, "./checkpoint-250")

# Merge and use the fine-tuned model
adapter_model = adapter_model.merge_and_unload()
adapter_model.eval()
```

## 📊 Generating Text with Fine-Tuned Model

Once the model is trained, you can use it to generate text:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="./checkpoint-250", tokenizer=tokenizer)
output = generator("The future of AI is", max_length=50)
print(output)
```

---

📧 **Contact:** ali.abdien.omar@gmail.com  

---


