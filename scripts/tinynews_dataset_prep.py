# prep_tiny_agnews.py
from datasets import load_dataset
import json, random
labels = ["World","Sports","Business","Sci/Tech"]
ds = load_dataset("ag_news")
train = ds["train"].shuffle(seed=42).select(range(200))
val   = ds["test"].shuffle(seed=42).select(range(50))

def fmt(x):
    text = x["text"].strip()
    y = labels[x["label"]]
    s = (
        "### Instruction:\n"
        "Classify the news into one of: World, Sports, Business, Sci/Tech.\n\n"
        "### Input:\n"
        f"{text}\n\n"
        "### Response:\n"
        f"{y}"
    )
    return {"text": s}

with open("train.jsonl","w",encoding="utf-8") as f:
    for r in train.map(fmt):
        f.write(json.dumps(r) + "\n")
with open("val.jsonl","w",encoding="utf-8") as f:
    for r in val.map(fmt):
        f.write(json.dumps(r) + "\n")
print("Wrote train.jsonl, val.jsonl")
