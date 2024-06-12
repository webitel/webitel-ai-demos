import csv
import gradio as gr
import pandas as pd
import logging
import numpy as np
import json
import re
import torch
import gc
from dotenv import load_dotenv
from src.chatbot import RagAgent, generate_questions
from src.vector_db import init_vector_db_and_get_retriever, init_ensemble_retriever
from langchain_core.messages import HumanMessage, AIMessage
from src.train_cross_encoder import train_cross_encoder
import os

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
data_folder = os.environ.get("DATA_FOLDER")
faq_file = os.environ.get("FAQ_FILE")
cross_encoder_file = os.environ.get("CROSS_ENCODER_FILE")
openai_model = os.environ.get("OPENAI_MODEL")

logname = "app.logs"
logging.basicConfig(
    filename=logname,
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger("UI")


retriever = init_ensemble_retriever(faq_file)
agent = RagAgent(vector_retriever=retriever)

context_info = []


# Function to handle user input and generate response
def chatbot_run(question, history_gradio):
    global context_info
    history = []

    if history_gradio == []:
        context_info = []
    for message_pair in history_gradio:
        history.append(HumanMessage(message_pair[0]))
        history.append(AIMessage(message_pair[1]))
    response, best_ranked_docs = agent.advanced_rag(question, chat_history=history)
    context_info.append((question, best_ranked_docs))
    return response

def add_to_vector_database(question, answer, autogenerated_questions, category):
    generated_questions = []
    if autogenerated_questions == "None":
        pass
    else:
        pattern = re.compile(r"Q\d+: (.+?)\n")
        matches = pattern.findall(autogenerated_questions)
        for match in matches:
            generated_questions.append(match)

    all_questions = [question] + generated_questions
    for q in all_questions:
        with open(faq_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f'"Питання : {q} Відповідь: {answer}"', category])
            logger.info(f'Added new example to vector database, "Вопрос : {q} Ответ: {answer}\. {category}"')

    # Reload retriever and agent
    global retriever, agent
    retriever = init_ensemble_retriever(faq_file)
    agent = RagAgent(vector_retriever=retriever)


def add_sample_for_cross_encoder(df: pd.DataFrame):
    query_data = {}
    for index, row in df.iterrows():
        query = str(row["Query"])
        positive = row["Positive Answer"]
        negative = row["Negative Answer"]

        if query not in query_data:
            query_data[query] = {"All_Positivies": [], "All_Negatives": []}

        if positive:
            query_data[query]["All_Positivies"].append(positive)
        if negative:
            query_data[query]["All_Negatives"].append(negative)

    try:
        current_data = pd.read_csv(cross_encoder_file)

        for index, row in current_data.iterrows():
            query = str(row["query"])
            positive = row["positive"]
            negative = row["negative"]

            if query not in query_data:
                query_data[query] = {"All_Positivies": [], "All_Negatives": []}

            if positive:
                query_data[query]["All_Positivies"].extend(json.loads(positive.replace("'", '"')))
            if negative:
                query_data[query]["All_Negatives"].extend(json.loads(negative.replace("'", '"')))
    except:
        logger.error("No data in cross encoder file")
        
    new_data = []
    for query, answers in query_data.items():
        new_data.append(
            {
                "query": query,
                "positive": list(set(answers["All_Positivies"])),
                "negative": list(set(answers["All_Negatives"])),
            }
        )

    new_df = pd.DataFrame(new_data)

    new_df.to_csv(cross_encoder_file, index=False)


def change_user(
    full_name, contract, phone, debt, overdue, model_name, user_category, startDate, dueDate
):
    global agent, retriever
    for ind_retrievers in retriever.retrievers:
        del ind_retrievers
    del retriever
    del agent.retriever, agent.reranker
    del agent
    gc.collect()
    retriever = init_ensemble_retriever(faq_file, category=user_category)
    agent = RagAgent(vector_retriever=retriever)
    agent.change_user(
        full_name, contract, overdue, phone, debt, user_category, startDate, dueDate
    )
    agent.change_reranker(model_name)
    torch.cuda.empty_cache()


def refresh_context():
    global context_info
    return pd.DataFrame(context_info, columns=["Question", "Documents"])


def refresh_data():
    global retriever, agent
    retriever = init_ensemble_retriever(faq_file)
    agent = RagAgent(vector_retriever=retriever)
    torch.cuda.empty_cache()
    return gr.Dataframe(pd.read_csv(faq_file), interactive=True)


def save_df(gr_df):
    gr_df.to_csv(faq_file, index=False)


with gr.Blocks() as demo:
    with gr.Tab("Settings"):
        name = gr.Textbox("Name", label="Name")
        contract = gr.Textbox("Contract", label="contract")
        phone = gr.Textbox("Phone", label="phone")
        debt = gr.Textbox("Debt", label="Debt")
        overdue = gr.Textbox("Overdue", label="Overdue")
        model_name = gr.Textbox("default", label="model_name")
        startDate = gr.Textbox("2022-01-01", label="startDate")
        dueDate = gr.Textbox("2022-01-10", label="dueDate")
        user_category = gr.Textbox("credit", label="user_category")

        btn = gr.Button("Change settings")

        btn.click(
            change_user,
            [
                name,
                contract,
                phone,
                debt,
                overdue,
                model_name,
                user_category,
                startDate,
                dueDate,
            ],
        )

    with gr.Tab("Chat"):
        gr.ChatInterface(
            chatbot_run,
            chatbot=gr.Chatbot(height=600, render=False, elem_id="chatbot"),
        )

    with gr.Tab("Add new example"):
        question = gr.Textbox("Question", label="Question")
        answer = gr.Textbox("Answer", label="Answer")
        category = gr.Textbox("Category", label="Category")
        btn = gr.Button("Add to Vector Database")
        autogenerated_questions = gr.Textbox("None", label="Autogenerated questions")
        generate_btn = gr.Button("Generate Questions")

        generate_btn.click(
            generate_questions,
            [question, answer],
            outputs=[autogenerated_questions],
        )
        btn.click(
            add_to_vector_database,
            [question, answer, autogenerated_questions, category],
        )

    with gr.Tab("Data"):
        dataset = pd.read_csv(faq_file)
        gr_df = gr.DataFrame(dataset, interactive=True)
        save_btn = gr.Button("Save")
        refresh_btn = gr.Button("Refresh")
        refresh_btn.click(
            refresh_data,
            outputs=[gr_df],
        )
        save_btn.click(
            save_df,
            [gr_df],
        )

    with gr.Tab("Context Used In Answers"):
        context_df = pd.DataFrame(context_info, columns=["Question", "Documents"])
        context_df_gradio = gr.DataFrame(context_df, interactive=False)

        refresh_btn = gr.Button("Refresh")
        refresh_btn.click(
            refresh_context,
            outputs=[context_df_gradio],
        )

    with gr.Tab("Cross Encoder"):
        table = gr.Dataframe(
            headers=["Query", "Positive Answer", "Negative Answer"],
            interactive=True,
        )

        model_name = gr.Textbox("model name", label="model_name")
        btn_process_samples = gr.Button("Process Samples")
        btn_train_cross_encoder = gr.Button("Train Encoder")

        def process_samples(table):
            table["Query"] = table["Query"].replace("", np.nan)
            table = table.dropna()
            print(table)
            add_sample_for_cross_encoder(table)

        btn_process_samples.click(
            process_samples,
            table,
        )
        btn_train_cross_encoder.click(
            train_cross_encoder,
            model_name,
        )

demo.queue().launch(share=True)
