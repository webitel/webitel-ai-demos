import logging
import os
import shutil
from io import BytesIO

import dotenv
import minio
import pandas as pd
from constants import default_reranker_file, model_save_path
from sentence_transformers import InputExample, LoggingHandler
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)

dotenv.load_dotenv()
cross_encoder_file = os.environ.get("CROSS_ENCODER_FILE")


def train_cross_encoder(model_save_name):
    client = minio.Minio(
        endpoint="localhost:9002",
        access_key="minioroot",
        secret_key="miniopassword",
        secure=False,
    )
    bucket_name = "chatbot-rag"  # os.environ.get("MINIO_DEFAULT_BUCKETS")

    csv_data = client.get_object(bucket_name, default_reranker_file).data
    df = pd.read_csv(BytesIO(csv_data))
    # model_save_path += model_save_name

    # First, we define the transformer model we want to fine-tune
    model_name = "DiTy/cross-encoder-russian-msmarco"
    train_batch_size = 32
    num_epochs = 50

    # We set num_labels=1, which predicts a continuous score between 0 and 1
    model = CrossEncoder(model_name, num_labels=1, max_length=512)

    dev_samples = {}
    for i, row in df.iterrows():
        positive = row["positive"]
        negative = row["negative"]
        dev_samples[i] = {
            "query": row["query"],
            "positive": positive,
            "negative": negative,
        }

    train_samples = []

    for key, value in dev_samples.items():
        query = value["query"]
        positive_examples = value["positive"]
        negative_examples = value["negative"]

        for pos_example in positive_examples:
            train_samples.append(InputExample(texts=[query, pos_example], label=1))

        for neg_example in negative_examples:
            train_samples.append(InputExample(texts=[query, neg_example], label=0))

    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=train_batch_size
    )

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
        # output_path=model_save_path,
        use_amp=True,
        save_best_model=False,
    )

    # Save latest model
    model.save(model_save_name)
    for file_path in os.listdir(model_save_name):
        client.fput_object(
            bucket_name,
            f"{model_save_path}/{model_save_name}/{file_path}",
            os.path.join(model_save_name, file_path),
        )
    shutil.rmtree(model_save_name)
    return "Model trained successfully!"


if __name__ == "__main__":
    train_cross_encoder(model_save_name="test_model")
