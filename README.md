# Phi-3 Mini Fine-tuning with QLoRA

This project demonstrates parameter-efficient fine-tuning (QLoRA) of `microsoft/Phi-3-mini-4k-instruct` on the [AG News dataset](https://huggingface.co/datasets/ag_news).

## 🚀 Features
- Loads a large language model in 4-bit quantization with `bitsandbytes`
- Prepares a tiny instruction-formatted AG News dataset
- Fine-tunes using `LoRA` adapters
- Runs inference with base model + adapters