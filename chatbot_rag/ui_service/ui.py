import ast
import os
import re
import requests
import io
import logging
import tempfile

import gradio as gr
import grpc
import numpy as np
import pandas as pd
import vector_db_pb2
import vector_db_pb2_grpc
from constants import default_reranker_file
from train_cross_encoder import train_cross_encoder
from utils import (
    add_sample_for_cross_encoder,
    generate_questions,
    get_samples_from_minio,
    refresh_data_context,
    remove_file_from_minio,
)

minio_login = os.environ.get("MINIO_ROOT_USER")
minio_password = os.environ.get("MINIO_ROOT_PASSWORD")
minio_bucket_name = os.environ.get("MINIO_DEFAULT_BUCKETS")
minio_url = os.environ.get("MINIO_URL")
VECTOR_DB_INTERFACE_HOST = os.environ.get("VECTOR_DB_INTERFACE_HOST")
VECTOR_DB_INTERFACE_PORT = os.environ.get("VECTOR_DB_INTERFACE_PORT")
TRANSLATION_HOST = os.environ.get("TRANSLATION_HOST")
TRANSLATION_PORT = os.environ.get("TRANSLATION_PORT")

channel = grpc.insecure_channel(
    f"{VECTOR_DB_INTERFACE_HOST}:{VECTOR_DB_INTERFACE_PORT}"
)
stub = vector_db_pb2_grpc.VectorDBServiceStub(channel)


def translate_text(text):
    response = requests.post(
        f"http://{TRANSLATION_HOST}:{TRANSLATION_PORT}/translate", json={"text": text}
    )
    return response.json()["translation"]


def add_to_vector_database(question, answer, autogenerated_questions, category):
    try:
        generated_questions = []
        if autogenerated_questions == "None":
            pass
        else:
            pattern = re.compile(r"Q\d+: (.+?)\n")
            matches = pattern.findall(autogenerated_questions)
            for match in matches:
                generated_questions.append(match)

        all_questions = [question] + generated_questions
        articles = []
        for q in all_questions:
            q_a = f"Питання: {q} Відповідь: {answer}"
            articles.append(
                vector_db_pb2.Article(
                    content=q_a,
                    categories=[category],
                )
            )
        response = stub.AddArticles(vector_db_pb2.AddArticlesRequest(articles=articles))
        ids, status = response.id, response.response_message
        gr.Info(message=status)
        gr.Info(message="Article ids: " + str(ids))
    except Exception as e:
        gr.Info(message="Error: " + str(e))


def refresh_data():
    try:
        articles = stub.GetArticles(vector_db_pb2.GetArticlesRequest()).articles
        dict_data = []
        for article in articles:
            current_data = {}
            current_data["id"] = article.id
            current_data["content"] = article.content
            current_data["categories"] = str(article.categories)
            dict_data.append(current_data)
        dataset = pd.DataFrame.from_dict(dict_data)
        gr_df = gr.DataFrame(dataset, interactive=True)
        gr.Info(message="Data refreshed")
        return gr_df
    except Exception as e:
        gr.Info(message="Error: " + str(e))


def save_df(gr_df):
    try:
        articles = []
        for i in range(gr_df.shape[0]):
            row = gr_df.iloc[i]
            print(row["categories"], type(row["categories"]))
            article = vector_db_pb2.Article(
                id=row["id"],
                content=row["content"],
                categories=ast.literal_eval(row["categories"]),
            )
            articles.append(article)
        message = stub.UpdateArticles(
            vector_db_pb2.UpdateArticlesRequest(articles=articles)
        )
        gr.Info(message=f"Data saved, {message}")
    except Exception as e:
        gr.Info(message="Error: " + str(e))


def upload_csv(
    file, overwrite=False, translate=False, progress=gr.Progress(track_tqdm=True)
):
    progress(0, desc="Uploading data...")
    gr.Info(message="Uploading data")
    df = pd.read_csv(io.BytesIO(file))
    for default_columns in ["question", "answer", "categories"]:
        if default_columns not in df.columns:
            gr.Info(message=f"Can not Load, Column {default_columns} is required")
            return
    if overwrite:
        articles = stub.GetArticles(vector_db_pb2.GetArticlesRequest()).articles
        uuid_to_remove = [article.id for article in articles]
        logging.debug(f"Removing articles with ids: {uuid_to_remove}")
        stub.RemoveArticles(vector_db_pb2.RemoveArticlesRequest(id=uuid_to_remove))

    articles = []
    for i in progress.tqdm(range(df.shape[0])):
        row = df.iloc[i]

        q_a = f"Питання: {row['question']} Відповідь: {row['answer']}"
        articles.append(
            vector_db_pb2.Article(
                content=q_a,
                categories=[row["categories"]],
            )
        )
        if translate:
            translated_question = translate_text(row["question"])
            q_a = f"Питання: {translated_question} Відповідь: {row['answer']}"
            articles.append(
                vector_db_pb2.Article(
                    content=q_a,
                    categories=[row["categories"]],
                )
            )

    stub.AddArticles(vector_db_pb2.AddArticlesRequest(articles=articles))
    gr.Info(message="Data uploaded")


def refresh_data_cross():
    try:
        df = get_samples_from_minio()
        gr_df_cross = gr.DataFrame(df, interactive=True)
        gr.Info(message="Data refreshed")
        return gr_df_cross
    except Exception as e:
        gr.Info(message="Error: " + str(e))


def download_csv(dataframe):
    dataframe_copy = dataframe.drop(columns=["content", "id"])
    dataframe_copy["question"] = dataframe["content"].apply(
        lambda x: (
            lambda y: y.split(":")[1].replace("Відповідь", "").strip()
            if len(y.split(":")) > 1
            else None
        )(x)
    )
    dataframe_copy["answer"] = dataframe["content"].apply(
        lambda x: (
            lambda y: "".join(y.split(":")[2:]).strip()
            if len(y.split(":")) > 2
            else None
        )(x)
    )
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    dataframe_copy.to_csv(temp_file.name, index=False)
    return temp_file.name


def upload_csv_cross(file, overwrite=False):
    df = pd.read_csv(io.BytesIO(file))
    for default_columns in ["query", "positive", "negative"]:
        if default_columns not in df.columns:
            gr.Info(message=f"Can not Load, Column {default_columns} is required")
            return
    if overwrite:
        remove_file_from_minio(file_name=default_reranker_file)
        gr.Info(message="Previous data removed")
    res = add_sample_for_cross_encoder(df)
    if res == "Samples added successfully!":
        gr.Info(message="Data uploaded")


def download_csv_cross(dataframe):
    df = get_samples_from_minio()
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(temp_file.name, index=False)
    return temp_file.name


with gr.Blocks() as demo:
    with gr.Tab("Add new example"):
        question = gr.Textbox("Question", label="Question")
        answer = gr.Textbox("Answer", label="Answer")
        category = gr.Textbox("Category", label="Category")
        btn = gr.Button("Add to Vector Database")
        autogenerated_questions = gr.Textbox("None", label="Autogenerated questions")
        generate_btn = gr.Button("Generate Questions")
        generate_and_translate = gr.Button("Generate and Translate")
        generate_btn.click(
            generate_questions,
            [question, answer, gr.Number(0, visible=False)],
            outputs=[autogenerated_questions],
            show_progress=True,
        )
        generate_and_translate.click(
            generate_questions,
            [question, answer, gr.Number(1, visible=False)],
            outputs=[autogenerated_questions],
            show_progress=True,
        )
        btn.click(
            add_to_vector_database,
            [question, answer, autogenerated_questions, category],
            show_progress=True,
        )

    with gr.Tab("Data"):
        articles = stub.GetArticles(vector_db_pb2.GetArticlesRequest()).articles
        dict_data = []
        for article in articles:
            current_data = {}
            current_data["id"] = article.id
            current_data["content"] = article.content
            current_data["categories"] = str(article.categories)
            dict_data.append(current_data)
        dataset = pd.DataFrame.from_dict(dict_data)
        gr_df = gr.DataFrame(dataset, interactive=True)
        with gr.Row():
            save_btn = gr.Button("Save")
            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(refresh_data, outputs=[gr_df], show_progress=True)
            save_btn.click(save_df, [gr_df], show_progress=True)
        with gr.Row():
            u = gr.UploadButton("Upload CSV", file_count="single", type="binary")
            checkbox = gr.Checkbox(label="Overwrite", value=False)
            translate = gr.Checkbox(label="Translate", value=False)
            u.upload(upload_csv, [u, checkbox, translate])
        with gr.Row():
            download_button = gr.Button("Download CSV")
            download_button.click(download_csv, [gr_df], outputs=gr.File())
    # with gr.Tab("Context Used In Answers"):
    #     context_df = pd.DataFrame(context_info, columns=["Question", "Documents"])
    #     context_df_gradio = gr.DataFrame(context_df, interactive=False)

    #     refresh_btn = gr.Button("Refresh")
    #     refresh_btn.click(
    #         refresh_context,
    #         outputs=[context_df_gradio],
    #     )

    with gr.Tab("Cross Encoder"):
        table = gr.Dataframe(
            headers=["query", "positive", "negative"],
            interactive=True,
        )

        model_name = gr.Textbox("model name", label="model_name")
        btn_process_samples = gr.Button("Process Samples")
        btn_train_cross_encoder = gr.Button("Train Encoder")
        refresh_btn_cross = gr.Button("Refresh")
        refresh_btn_cross.click(refresh_data_cross, outputs=[table], show_progress=True)

        def process_samples(table):
            try:
                table["query"] = table["query"].replace("", np.nan)
                table = table.dropna()
                print(table)
                res = add_sample_for_cross_encoder(table)
                gr.Info(message=res)
            except Exception as e:
                gr.Info(message="Error: " + str(e))

        btn_process_samples.click(
            process_samples,
            table,
        )

        def train_cross_encoder_with_popup(model_name):
            gr.Info(message="Started Training, please wait...")
            try:
                res = train_cross_encoder(model_name)
                gr.Info(message=res)
            except Exception as e:
                gr.Info(message="Error: " + str(e))

        btn_train_cross_encoder.click(
            train_cross_encoder_with_popup, model_name, show_progress=True
        )
        with gr.Row():
            u = gr.UploadButton("Upload CSV", file_count="single", type="binary")
            checkbox = gr.Checkbox(label="Overwrite", value=False)
            u.upload(upload_csv_cross, [u, checkbox])
        with gr.Row():
            download_button = gr.Button("Download CSV")
            download_button.click(download_csv_cross, [table], outputs=gr.File())

    with gr.Tab("Used Context"):
        gr_df_cross = gr.DataFrame(pd.DataFrame(), interactive=True)
        refresh_btn_context = gr.Button("Refresh")
        refresh_btn_context.click(
            refresh_data_context, outputs=[gr_df_cross], show_progress=True
        )

demo.queue().launch(server_name="0.0.0.0", share=False)
