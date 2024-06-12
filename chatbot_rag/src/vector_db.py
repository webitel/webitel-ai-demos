from langchain import hub
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import torch
import os
import dotenv

dotenv.load_dotenv()

device = os.getenv("DEVICE")
print("DEVICE VECTOR", device)

def init_vector_db_and_get_retriever(
    csv_path, model_name="ai-forever/sbert_large_nlu_ru"
):
    # if path == "./data/webitel_dataset.csv":
    #     loader = CSVLoader(path)
    #     syntetic_loader = CSVLoader("./data/question_answers_syntetic.csv")
    #     syntetic_docs = syntetic_loader.load()
    #     docs = loader.load()
    #     docs.extend(syntetic_docs)
    # else:
    loader = CSVLoader(csv_path)
    docs = loader.load()

    embeding_model = HuggingFaceEmbeddings(
        model_name="ai-forever/sbert_large_nlu_ru", device = device
    )  # "ai-forever/ruBert-large")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeding_model)
    vectorstore.delete_collection()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeding_model)

    retriever = vectorstore.as_retriever(k=30)
    return retriever


def init_ensemble_retriever(
    csv_path, category="credit"
):
    loader = CSVLoader(csv_path, metadata_columns=["category"])
    docs = loader.load()
    if category != "all":
        temp_docs = []
        for doc in docs:
            if doc.metadata["category"] == category:
                temp_docs.append(doc)
        docs = temp_docs
    embeding_model = HuggingFaceEmbeddings(
        model_name="ai-forever/sbert_large_nlu_ru", model_kwargs = {"device": device}
    )  # "ai-forever/ruBert-large")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeding_model)
    # Retrieve and generate using the relevant snippets of the blog.
    embedding_retriever = vectorstore.as_retriever(k=30)

    bm25_retriever = BM25Retriever.from_documents(splits)
    bm25_retriever.k = 30
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, embedding_retriever], weights=[0.5, 0.5]
    )
    torch.cuda.empty_cache()

    return ensemble_retriever
