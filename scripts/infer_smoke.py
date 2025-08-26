from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base_id = "microsoft/Phi-3-mini-4k-instruct"
adapters = "phi3mini-agnews-smoke-adapters"

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype="bfloat16"
)

tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(base_id, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, adapters)

prompt = (
    "### Instruction:\nClassify the news into one of: World, Sports, Business, Sci/Tech.\n\n"
    "### Input:\nIndia's benchmark indices rallied as IT and banking stocks gained after the policy announcement.\n\n"
    "### Response:\n"
)
x = tok(prompt, return_tensors="pt").to(model.device)
y = model.generate(**x, max_new_tokens=8, do_sample=False,
                   eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
print(tok.decode(y[0], skip_special_tokens=True))
