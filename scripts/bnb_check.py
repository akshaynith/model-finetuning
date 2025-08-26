import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

m = "microsoft/Phi-3-mini-4k-instruct"
tok = AutoTokenizer.from_pretrained(m, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    m, load_in_4bit=True, device_map="auto"
)

print("Loaded 4-bit model on:", model.device)