from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

model_id = "microsoft/Phi-3-mini-4k-instruct"

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype="bfloat16"
)

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb, device_map="auto"
)

lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
model = get_peft_model(base, lora)

train_ds = load_dataset("json", data_files="train.jsonl")["train"]
val_ds   = load_dataset("json", data_files="val.jsonl")["train"]

sft_config = SFTConfig(
    output_dir="phi3mini-agnews-smoke",
    dataset_text_field="text",
    max_length=512,
    packing=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=24,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    eval_steps=50,
    save_steps=999999,
    bf16=True,
    fp16=False,
    gradient_checkpointing=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    # tokenizer=tok,
    args=sft_config
)
trainer.train()
trainer.save_model("phi3mini-agnews-smoke-adapters")
print("Saved adapters to phi3mini-agnews-smoke-adapters")
