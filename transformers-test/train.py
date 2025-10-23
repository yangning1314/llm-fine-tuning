from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding
'''
导入数据集
'''
raw_datasets = load_from_disk("data")
checkpoint = "directory_on_my_computer"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

'''
将特征添加到数据集中
'''
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
'''
用于训练和评估的所有超参数。您必须提供的唯一参数是保存训练模型的目录，以及沿途的检查点。对于所有其他内容，您可以保留默认值
'''
from transformers import TrainingArguments

training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    fp16=True,  # Enable mixed precision
)

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)


from transformers import Trainer

import evaluate


def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()


predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)

import numpy as np

preds = np.argmax(predictions.predictions, axis=-1)


metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)




