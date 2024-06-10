import gzip
import logging
import os
import tarfile
from datetime import datetime

import tqdm
from torch.utils.data import DataLoader
import pandas as pd
from sentence_transformers import InputExample, LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
import json

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

def train_cross_encoder(model_path):
    df = pd.read_csv("client_data/cross_encoder_data.csv")
    # First, we define the transformer model we want to fine-tune
    model_name = "DiTy/cross-encoder-russian-msmarco"
    train_batch_size = 32
    num_epochs = 50
    model_save_path = (
        f"output/{model_path}"
    )

    # We set num_labels=1, which predicts a continuous score between 0 and 1
    model = CrossEncoder(model_name, num_labels=1, max_length=512)

    dev_samples = {}
    for i,row in df.iterrows():
        positive = row["positive"]
        negative = row["negative"]
        dev_samples[i] = {"query": row["query"], "positive": json.loads(positive.replace("'",'"')), "negative": json.loads(negative.replace("'",'"'))}

    train_samples = []

    for key, value in dev_samples.items():
        query = value["query"]
        positive_examples = value["positive"]
        negative_examples = value["negative"]
        
        for pos_example in positive_examples:
            train_samples.append(InputExample(texts=[query, pos_example], label=1))
        
        for neg_example in negative_examples:
            train_samples.append(InputExample(texts=[query, neg_example], label=0))

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    evaluator = CERerankingEvaluator(dev_samples, name="train-eval")

    # Configure the training
    warmup_steps = 10
    logging.info("Warmup-steps: {}".format(warmup_steps))


    # Train the model
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        evaluation_steps=2,
        warmup_steps=warmup_steps,
        output_path=model_save_path,
        use_amp=True,
    )

    # Save latest model
    model.save(model_save_path + "-latest")
    
if __name__ == "__main__":
    train_cross_encoder()