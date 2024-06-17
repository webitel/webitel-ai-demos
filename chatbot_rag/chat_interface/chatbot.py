import weaviate
import numpy as np

from langchain_weaviate import WeaviateVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import os
from utils import load_prompt



openai_model = os.environ.get("OPENAI_MODEL")
device = os.environ.get("DEVICE")
db_host = os.environ.get("HOST")
http_port = int(os.environ.get("HTTP_PORT"))
grpc_port = int(os.environ.get("GRPC_PORT"))

class Reranker():
    def __init__(self, model_name="DiTy/cross-encoder-russian-msmarco"):
        self.model = CrossEncoder(model_name, max_length=512, device=device)#
    
    def get_rank(self, queries, docs):
        return self.model.rank(queries, docs)
    
class ChatBot():
    def __init__(self, reranker_model = "DiTy/cross-encoder-russian-msmarco"):
        collection_name = "KnowledgeBase"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="ai-forever/sbert_large_nlu_ru", model_kwargs = {"device": device}
        )
        self.reranker = Reranker(reranker_model)
        self.query_analyzer = create_query_analyzer()
        self.qa_chain = create_question_answer_chain()
        
        print(db_host)
        print(http_port)
        print(grpc_port)
        self.client = weaviate.WeaviateClient(connection_params=weaviate.connect.ConnectionParams(
            http=weaviate.connect.ProtocolParams(host=db_host,port=http_port,secure = False),
            grpc=weaviate.connect.ProtocolParams(host=db_host,port=grpc_port,secure = False)
            ))
        self.client.connect()
        self.retriever = WeaviateVectorStore(self.client, attributes=['categories'], text_key = 'content', index_name=collection_name, embedding=self.embedding_model)
        self.client.close()

    def retrieve(self,query):
        vector = self.embedding_model.embed_documents([query])[0]
        kwargs= {'return_uuids':True,
                'vector':vector,
                'alpha':0.5 # 1 - pure vector search, 0 - pure keyword search
                }
        return self.retriever.similarity_search(query=query,k=10,**kwargs)
    
    def answer(self, input_query, chat_history, **kwargs):
        self.client.connect()
        
        self.qa_chain = create_question_answer_chain(**kwargs)
        
        
        #1. generate new queries 
        new_queries = self.query_analyzer.invoke({"question":input_query})
        new_queries.append(ParaphrasedQuery(paraphrased_query=input_query))
        
        #2. retrieve documents
        context_docs = []
        for query in new_queries:
            # context_docs.extend(self.history_aware_retriever.invoke({"input":prefix_query + query.paraphrased_query,"chat_history":chat_history}))
            # context_docs.extend(self.history_aware_retriever.invoke({"input":query.paraphrased_query,"chat_history":chat_history}))
            context_docs.extend(self.retrieve(query.paraphrased_query))
            
        #3. remove duplicates (at least partially)
        deduped_docs = []
        for doc in context_docs:
            if doc not in deduped_docs:
                deduped_docs.append(doc)    
        
        #3. rerank documents and select best 5 
        ranks = self.reranker.get_rank(input_query, [doc.page_content for doc in deduped_docs])
        best_ranked_docs = []
        for rank in ranks[:5]:
            best_ranked_docs.append(deduped_docs[rank['corpus_id']])
        
        answer = self.qa_chain.invoke({"input":input_query,"context":best_ranked_docs, "chat_history":chat_history})
        
        self.client.close()
        return answer,best_ranked_docs
    

class ParaphrasedQuery(BaseModel):
    """You have performed query expansion to generate a paraphrasing of a question."""

    paraphrased_query: str = Field(
        ...,
        description="A unique paraphrasing of the original question.",
    )
    

def create_query_analyzer():
    system = load_prompt('prompts/query_analyzer_prompt.txt')
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    llm = ChatOpenAI(model=openai_model, temperature=0,)
    llm_with_tools = llm.bind_tools([ParaphrasedQuery])
    query_analyzer = prompt | llm_with_tools | PydanticToolsParser(tools=[ParaphrasedQuery])
    
    return query_analyzer

def create_question_answer_chain(**kwargs):
    llm = ChatOpenAI(model=openai_model,temperature=0.4)
    prefix = load_prompt('prompts/question_answer_chain.txt')
    client_info = """You may answer about this additional data about the client, without context, but you should use this data to provide more accurate answer. 
    Here is additional data about the client you are speaking with:"""
    for kwargs_key, kwargs_value in kwargs.items():
        client_info = client_info + f"\n {kwargs_key} : {kwargs_value}"
    client_info += "\n"

    
    postfix = """Here is the context that you can use to answer the question and provide accrurate answer, you can rephrase answer from the context to make it more natural and accurate.:
    {context}"""
    
    qa_system_prompt = prefix + client_info + postfix
    
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
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
    
    answer,best_ranked_docs = bot.answer("Як мене звати??",[], **{"client_name":"Вася Пупкін"})
    print(answer, best_ranked_docs)
    history = ["Яка столиця франції?",answer]

    answer,best_ranked_docs = bot.answer("Яке моє минуле питання?",history, **{"client_name":"Вася Пупкін"})
    print(answer, best_ranked_docs)
    
    
    
    # bot.answer("What is the capital of France?")