import weaviate
from langchain_community.retrievers import (
    WeaviateHybridSearchRetriever,
)
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

class ChatBot():
    def __init__(self,connection_uri):
        collection_name = "KnowledgeBase"
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="ai-forever/sbert_large_nlu_ru", model_kwargs = {"device": 'cpu'}
        )
        
        client = weaviate.Client(url=connection_uri)
            
        self.retriever = WeaviateHybridSearchRetriever(
            client=client,
            index_name=collection_name,
            text_key="content",
            attributes=[],
            create_schema_if_missing=False,
            alpha=0,# less -> more weight to bm25, more -> more weight to dense
            k=20
        )
    
    def retrieve(self,query):
        vector = self.embedding_model.embed_documents([query])[0]
        return self.retriever.get_relevant_documents(query, hybrid_search_kwargs={"vector": vector} )
    
    def answer(self, question):
        raise NotImplementedError("Method not implemented!")
    
if __name__ == "__main__":
    bot = ChatBot("http://localhost:9000")
    res = bot.retrieve("Привіт")
    print(res)
    # bot.answer("What is the capital of France?")