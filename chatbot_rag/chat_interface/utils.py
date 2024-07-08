import minio
from sentence_transformers import CrossEncoder
import os
from constants import reranker_models_path
import shutil
import logging
import pandas as pd
import io


def load_prompt(prompt_path):
    with open(prompt_path, "r") as f:
        prompt = f.read()
    return prompt


def load_model(model_name, device, bucket_name, access_key, secret_key, minio_url):
    if os.path.exists(os.path.join(reranker_models_path, model_name)):
        return CrossEncoder(model_name, max_length=512, device=device)

    client = minio.Minio(
        minio_url, access_key=access_key, secret_key=secret_key, secure=False
    )

    if (
        os.path.exists(reranker_models_path)
        and len(os.listdir(reranker_models_path)) > 3
    ):
        ## remove oldest folder
        dirs = [_dir for _dir in os.listdir(reranker_models_path)]
        sorted_dirs = list(
            sorted(
                dirs,
                key=lambda x: os.stat(os.path.join(reranker_models_path, x)).st_mtime,
            )
        )
        print(f"removing {sorted_dirs[-1]}")
        logging.error(f"removing {sorted_dirs[-1]}")
        shutil.rmtree(os.path.join(reranker_models_path, sorted_dirs[-1]))

    for item in client.list_objects(
        bucket_name,
        prefix=os.path.join(reranker_models_path, model_name),
        recursive=True,
    ):
        client.fget_object(bucket_name, item.object_name, item.object_name)

    return CrossEncoder(
        os.path.join(reranker_models_path, model_name), max_length=512, device=device
    )


def store_context_info_to_minio(
    question, answer, best_ranked_docs, bucket_name, access_key, secret_key, minio_url
):
    client = minio.Minio(
        minio_url, access_key=access_key, secret_key=secret_key, secure=False
    )

    # Ensure the bucket exists
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)

    all_files = [
        item.object_name for item in client.list_objects(bucket_name, recursive=True)
    ]

    # Create context.csv if it doesn't exist
    if "context.csv" not in all_files:
        context_df = pd.DataFrame(
            columns=["question", "answer", "docs_content", "docs_ids"]
        )
        context_df.to_csv("context.csv", index=False)
        client.fput_object(bucket_name, "context.csv", "context.csv")

    # Get the context.csv file from minio
    context_file = client.get_object(bucket_name, "context.csv")
    context_file_bytes = context_file.read()

    # Read as pandas DataFrame
    context_df = pd.read_csv(io.BytesIO(context_file_bytes))

    # Prepare the new context info
    docs_content = [doc.page_content for doc in best_ranked_docs]
    docs_ids = [doc.metadata.get("uuid", None) for doc in best_ranked_docs]
    new_entry = pd.DataFrame(
        {
            "question": [question],
            "answer": [answer],
            "docs_content": [docs_content],
            "docs_ids": [docs_ids],
        }
    )

    # Append the new context info to the DataFrame
    context_df = pd.concat([context_df, new_entry], ignore_index=True)

    # Save the updated context_df to a temporary CSV file
    context_df.to_csv("updated_context.csv", index=False)

    # Upload the updated CSV file to Minio
    client.fput_object(bucket_name, "context.csv", "updated_context.csv")


if __name__ == "__main__":
    # model_name = "DiTy/cross-encoder-russian-msmarco"
    # device = "cuda"
    # minio_url = 'localhost:9002'
    model = load_model(
        "test_model",
        "cpu",
        "chatbot-rag",
        "minioroot",
        "miniopassword",
        "localhost:9002",
    )
    print(model.predict(["привет"]))
