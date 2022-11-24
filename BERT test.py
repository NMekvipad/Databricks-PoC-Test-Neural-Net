# Databricks notebook source
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


metric = evaluate.load("accuracy")
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")
dataset = load_dataset("tweet_eval", 'sentiment')
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
model = AutoModelForSequenceClassification.from_pretrained("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", num_labels=20)

# COMMAND ----------

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# COMMAND ----------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# COMMAND ----------

trainer.train()

# COMMAND ----------

model.save_pretrained("pre_trained_model/tuned_bert_model")
