# model.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax
from datasets import Dataset, DatasetDict
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
import os

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def preprocess(text):
    return text


# Function for inference
def inference(text):
    text = preprocess(text)

    # Tokenize input
    encoded_input = tokenizer(text, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

    # Print labels and scores
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    results = []
    for i in range(scores.shape[0]):
        label = config.id2label[ranking[i]]
        score = scores[ranking[i]]
        results.append({"label": label, "score": np.round(float(score), 4)})

    return results


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def train_model(dataset_json, model_name):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # Load dataset from JSON
    dataset = Dataset.from_dict(dataset_json)

    # Tokenize dataset
    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Split the dataset into training and validation sets
    train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
    datasets = DatasetDict(
        {"train": train_test_split["train"], "validation": train_test_split["test"]}
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize Trainer with compute_metrics argument
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,  # Pass tokenizer for collation
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    save_directory = f"./saved_models/{model_name}"
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    # Evaluate the model
    eval_results = trainer.evaluate()
    return eval_results
