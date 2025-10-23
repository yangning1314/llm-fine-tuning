import torch
from torch.optim import AdamW
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from datasets import load_dataset
from transformers import DataCollatorWithPadding



checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
raw_datasets = load_dataset("glue", "mrpc")

def tokenize_function(examples):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(raw_datasets['train']['sentence1'])

# sequences = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "This course is amazing!",
# ]
#
# batch = tokenizer(sequences,padding = True,truncation = True, return_tensors="pt")
#
# batch['labels'] = torch.tensor([1,1])
#
# optimizer = AdamW(model.parameters())
#
# loss = model(**batch).loss
# loss.backward()
# optimizer.step()









