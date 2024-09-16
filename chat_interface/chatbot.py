import contextlib
import os
import requests
from loguru import logger
import re

import weaviate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.output_parsers import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from sentence_transformers import CrossEncoder
from utils import load_model, load_prompt, store_context_info_to_minio
from weaviate.classes.query import Filter


@contextlib.contextmanager
def timeout_context_manager(context):
    """Context manager to add timeout to a block of code"""
    try:
        yield
    finally:
        if context.time_remaining() <= 0:
            raise TimeoutError("Timeout")


openai_model = os.environ.get("OPENAI_MODEL")
device = "cpu"  # os.environ.get("DEVICE") currently the model is only on cpu
db_host = os.environ.get("HOST")
http_port = int(os.environ.get("HTTP_PORT"))
grpc_port = int(os.environ.get("GRPC_PORT"))
minio_root_user = os.environ.get("MINIO_ROOT_USER")
minio_root_password = os.environ.get("MINIO_ROOT_PASSWORD")
minio_default_bucket = os.environ.get("MINIO_DEFAULT_BUCKETS")
minio_url = os.environ.get("MINIO_URL")
emedding_service_url = os.environ.get("EMBEDDING_SERVICE_URL")


class Reranker:
    def __init__(self, model_name="DiTy/cross-encoder-russian-msmarco"):
        if (
            model_name == "DiTy/cross-encoder-russian-msmarco"
            or model_name == "default"
            or model_name == ""
        ):
            self.model = CrossEncoder(
                "DiTy/cross-encoder-russian-msmarco", max_length=512, device=device
            )
            logger.debug("Loaded default model")
        else:
            self.model = load_model(
                model_name,
                device,
                minio_default_bucket,
                minio_root_user,
                minio_root_password,
                minio_url,
            )
            logger.debug(f"Loaded model: {model_name} from minio")

    def get_rank(self, queries, docs):
        logger.debug(f"Ranking {queries} queries and {docs} documents")
        return self.model.rank(queries, docs)


class ChatBot:
    def __init__(self, reranker_model="DiTy/cross-encoder-russian-msmarco"):
        collection_name = "KnowledgeBase"
        self.reranker = Reranker(reranker_model)
        self.reranker_name = reranker_model
        self.query_analyzer = create_query_analyzer()
        self.qa_chain = create_question_answer_chain()

        self.client = weaviate.WeaviateClient(
            connection_params=weaviate.connect.ConnectionParams(
                http=weaviate.connect.ProtocolParams(
                    host=db_host, port=http_port, secure=False
                ),
                grpc=weaviate.connect.ProtocolParams(
                    host=db_host, port=grpc_port, secure=False
                ),
            )
        )
        self.client.connect()
        self.retriever = WeaviateVectorStore(
            self.client,
            attributes=["categories"],
            text_key="content",
            index_name=collection_name,
            embedding=self.get_embedding,
        )

    def get_embedding(self, text):
        logger.debug(f"Getting embedding for: {text}")
        response = requests.post(emedding_service_url, json={"text": text})
        vector = response.json()["embedding"]
        return vector

    def retrieve(self, query, categories, k=10):
        logger.debug(
            f"Retrieving documents for query: {query} and categories: {categories}. k={k}"
        )
        vector = self.get_embedding(query)
        if "all" in categories:
            kwargs = {
                "return_uuids": True,
                "vector": vector,
                "alpha": 0.5,  # 1 - pure vector search, 0 - pure keyword search,
            }
        else:
            kwargs = {
                "return_uuids": True,
                "vector": vector,
                "alpha": 0.5,  # 1 - pure vector search, 0 - pure keyword search,
                "filters": Filter.by_property("categories").contains_all(categories),
                # TODO category ['ans] will match 'answer', 'answers','ans' etc. Probably need to fix it
            }
        return self.retriever.similarity_search(query=query, k=k, **kwargs)

    def post_process_answer(self, answer):
        answer = answer.replace("{", "")
        answer = answer.replace("}", "")
        return answer

    def answer(self, input_query, chat_history, categories, context, **kwargs):
        try:
            with timeout_context_manager(context):
                self.qa_chain = create_question_answer_chain(**kwargs)

                # if there are no categories reply without context
                if len(categories) == 0:
                    logger.debug("Answering without context")
                    answer = self.qa_chain.invoke(
                        {
                            "input": input_query,
                            "context": [],
                            "chat_history": chat_history,
                        }
                    )
                    return self.post_process_answer(answer), []

                logger.debug(f"Categories: {categories}")

            with timeout_context_manager(context):
                # 1. generate new queries
                new_queries = self.query_analyzer.invoke({"question": input_query})
                self.history_analyzer = create_query_analyzer()
                new_queries.append(ParaphrasedQuery(paraphrased_query=input_query))

                logger.debug(f"New queries: {new_queries}")

            with timeout_context_manager(context):
                # 2. retrieve documents
                context_docs = []
                for query in new_queries:
                    context_docs.extend(
                        self.retrieve(
                            query.paraphrased_query,
                            categories,
                            k=int(100 / len(new_queries)),
                        )
                    )

                logger.debug(f"Retrieved docs: {context_docs}")

                # 3. remove duplicates
                deduped_docs = []
                for doc in context_docs:
                    if doc not in deduped_docs:
                        deduped_docs.append(doc)

                # 4. transform kwargs
                transformed_kwargs = {
                    re.sub(r"\s*\(.*?\)\s*", "", key).strip(): value
                    for key, value in kwargs.items()
                }
                # Replace keys in doc.page_content with corresponding values
                for doc in deduped_docs:
                    page_content = doc.page_content
                    for key, value in transformed_kwargs.items():
                        page_content = page_content.replace(key, value)
                    doc.page_content = page_content

                logger.debug(f"Transformed docs: {deduped_docs}")
            with timeout_context_manager(context):
                if len(deduped_docs) != 0:
                    # 4. rerank documents and select best 10
                    ranks = self.reranker.get_rank(
                        input_query, [doc.page_content for doc in deduped_docs]
                    )
                    best_ranked_docs = [
                        deduped_docs[rank["corpus_id"]] for rank in ranks[:10]
                    ]

                    logger.debug(f"Used docs: {best_ranked_docs}")
                    answer = self.qa_chain.invoke(
                        {
                            "input": input_query,
                            "context": best_ranked_docs,
                            "chat_history": chat_history,
                        }
                    )
                else:
                    answer = self.qa_chain.invoke(
                        {
                            "input": input_query,
                            "context": [],
                            "chat_history": chat_history,
                        }
                    )
                answer = self.post_process_answer(answer)
                store_context_info_to_minio(
                    input_query,
                    answer,
                    best_ranked_docs,
                    minio_default_bucket,
                    minio_root_user,
                    minio_root_password,
                    minio_url,
                )
                return (
                    answer,
                    best_ranked_docs if "best_ranked_docs" in locals() else [],
                )

        except TimeoutError as e:
            logger.error(f"Timeout occurred in answer method: {str(e)}")
            raise e  # Propagate the TimeoutError to handle it appropriately at a higher level

        except Exception as e:
            logger.error(f"Exception occurred in answer method: {str(e)}")
            raise e  # Handle other exceptions according to your application's error handling strategy

    def reload_reranker_model(self, model_name):
        self.reranker = Reranker(model_name)
        self.reranker_name = model_name


class ParaphrasedQuery(BaseModel):
    """You have performed query expansion to generate a paraphrasing of a question."""

    paraphrased_query: str = Field(
        ...,
        description="A unique paraphrasing of the original question.",
    )


def create_history_query_analyzer():
    system = load_prompt("prompts/history_aware_retriever_prompt.txt")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
            ("chat_history", "{chat_history}"),
        ]
    )
    llm = ChatOpenAI(
        model=openai_model,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools([ParaphrasedQuery])
    query_analyzer = (
        prompt | llm_with_tools | PydanticToolsParser(tools=[ParaphrasedQuery])
    )

    return query_analyzer


def create_query_analyzer():
    system = load_prompt("prompts/query_analyzer_prompt.txt")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(
        model=openai_model,
        temperature=0,
    )
    llm_with_tools = llm.bind_tools([ParaphrasedQuery])
    query_analyzer = (
        prompt | llm_with_tools | PydanticToolsParser(tools=[ParaphrasedQuery])
    )

    return query_analyzer


def create_question_answer_chain(**kwargs):
    llm = ChatOpenAI(model=openai_model, temperature=0.4)
    qa_prompt_str = load_prompt("prompts/question_answer_chain.txt")
    client_info = ""
    for kwargs_key, kwargs_value in kwargs.items():
        client_info = client_info + f"\n {kwargs_key} : {kwargs_value}"
    client_info += "\n"

    qa_prompt_str = qa_prompt_str.replace("{user_metadata}", client_info)
    logger.debug(f"QA prompt: {qa_prompt_str}")
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_prompt_str),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return question_answer_chain


if __name__ == "__main__":
    bot = ChatBot("http://localhost:9000")
    # res = bot.retrieve("Привіт")
    # print(res)

    answer, best_ranked_docs = bot.answer(
        "Як мене звати??", [], **{"client_name": "Вася Пупкін"}
    )
    print(answer, best_ranked_docs)
    history = ["Яка столиця франції?", answer]

    answer, best_ranked_docs = bot.answer(
        "Яке моє минуле питання?", history, **{"client_name": "Вася Пупкін"}
    )
    print(answer, best_ranked_docs)

    # bot.answer("What is the capital of France?")
