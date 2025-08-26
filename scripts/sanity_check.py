from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

model_id = "microsoft/Phi-3-mini-4k-instruct"

# 4-bit quantized load (QLoRA path)
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16"  # change to "float16" if bf16 isn't supported
)

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto"
)

# Simple deterministic generation
prompt = "You are a helpful assistant. In one sentence, explain what QLoRA is."
inputs = tok(prompt, return_tensors="pt").to(model.device)
streamer = TextStreamer(tok)

out = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=False  # greedy for reproducibility
)

print(tok.decode(out[0], skip_special_tokens=True))
