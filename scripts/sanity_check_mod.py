from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_id = "microsoft/Phi-3-mini-4k-instruct"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype="bfloat16"  # use "float16" if bf16 isn't supported
)

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb,
    device_map="auto"
)

# Use chat template
messages = [
    {"role": "system", "content": "You are a concise, helpful AI assistant."},
    {"role": "user", "content": "In one sentence, explain what QLoRA is."}
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    max_new_tokens=64,
    do_sample=False,
    eos_token_id=tok.eos_token_id,
    pad_token_id=tok.pad_token_id
)
print(tok.decode(out[0], skip_special_tokens=True))
