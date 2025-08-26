from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "microsoft/Phi-3-mini-4k-instruct"  # swap to TinyLlama, Llama-3-3B, etc.

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # good default
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16" # if bf16 errors, change to "float16"
)

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto"
)

print("OK on:", model.device)
