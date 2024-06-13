
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
    utility
)

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from langchain_milvus.utils.sparse import BM25SparseEmbedding
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def get_next_id():
    if os.path.exists('last_id.txt'):
        with open('last_id.txt', 'r') as f:
            last_id = int(f.read())
    else:
        last_id = 0

    next_id = last_id + 1

    with open('last_id.txt', 'w') as f:
        f.write(str(next_id))

    return str(next_id)


# only_texts = [text for text,category in texts]

# dense_dim = len(dense_embedding_func.embed_query(only_texts[1]))
# # dense_dim

# sparse_embedding_func = BM25SparseEmbedding(corpus=only_texts)
# # sparse_embedding_func.embed_query(texts[1])


# pk_field = "doc_id"
# dense_field = "dense_vector"
# sparse_field = "sparse_vector"
# text_field = "text"
# category_field = "category"
# fields = [
#     FieldSchema(
#         name=pk_field,
#         dtype=DataType.VARCHAR,
#         is_primary=True,
#         auto_id=True,
#         max_length=100,
#     ),
#     FieldSchema(name=dense_field, dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
#     FieldSchema(name=sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR),
#     FieldSchema(name=text_field, dtype=DataType.VARCHAR, max_length=65_535),
#     FieldSchema(name=category_field, dtype=DataType.VARCHAR, max_length=128),
# ]

# schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
# collection = Collection(
#     name="IntroductionToTheNovels", schema=schema, consistency_level="Strong"
# )

# dense_index = {"index_type": "FLAT", "metric_type": "IP"}
# collection.create_index("dense_vector", dense_index)
# sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
# collection.create_index("sparse_vector", sparse_index)
# collection.flush()

# entities = []
# for text,category in texts:
#     entity = {
#         dense_field: dense_embedding_func.embed_documents([text])[0],
#         sparse_field: sparse_embedding_func.embed_documents([text])[0],
#         text_field: text,
#         category_field: category,
#     }
#     entities.append(entity)
# collection.insert(entities)
# collection.load()

# sparse_search_params = {"metric_type": "IP"}
# dense_search_params = {"metric_type": "IP", "params": {}}
# retriever = MilvusCollectionHybridSearchRetriever(
#     collection=collection,
#     rerank=WeightedRanker(0.5, 0.5),
#     anns_fields=[dense_field, sparse_field],
#     field_embeddings=[dense_embedding_func, sparse_embedding_func],
#     field_search_params=[dense_search_params, sparse_search_params],
#     top_k=3,
#     text_field=text_field,
# )

# retriever.invoke("What are the story about ventures?")



class VectorDatabase():
    def __init__(self,connection_uri="http://localhost:19530", collection_name ='milvus_collection3'):
        connections.connect(uri=connection_uri)
        if utility.has_collection(collection_name):
            self.collection = Collection(name=collection_name)
        else:
            self.__init_collection(collection_name)
        self.collection = Collection(name=collection_name)
        self.dense_embedding_func = HuggingFaceEmbeddings(
            model_name="ai-forever/sbert_large_nlu_ru", model_kwargs = {"device": 'cpu'}
        )
    
        self.collection.load()
        
    def __init_collection(self, collection_name):
        pk_field = "doc_id"
        dense_field = "dense_vector"
        sparse_field = "sparse_vector"
        text_field = "text"
        category_field = "category"
        dense_dim = 1024
        fields = [
            FieldSchema(
                name=pk_field,
                dtype=DataType.VARCHAR,
                is_primary=True,
                auto_id=False,
                max_length=100,
            ),
            FieldSchema(name=dense_field, dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
            FieldSchema(name=sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name=text_field, dtype=DataType.VARCHAR, max_length=65_535),
            FieldSchema(name=category_field, dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity = 15,max_length=512),
        ]

        schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
        collection = Collection(
            name=collection_name, schema=schema, consistency_level="Strong"
        )

        dense_index = {"index_type": "FLAT", "metric_type": "IP"}
        collection.create_index("dense_vector", dense_index)
        sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
        collection.create_index("sparse_vector", sparse_index)
        collection.flush()
    
    def __init_sparse_embedding(self, texts :list[str] = []):
        corpus = [entity['text'] for entity in self.collection.query(
            expr="doc_id != ''", 
            output_fields=["text"],
            consistency_level="Strong"
        )]
        if len(corpus) == 0 and texts:
            corpus = texts
            return BM25SparseEmbedding(corpus=corpus)
        elif len(corpus) > 0:
            print(corpus)
            corpus.extend(texts)
            return BM25SparseEmbedding(corpus=corpus)
        # logging.error("No corpus found")
        return None
    
    def insert(self, texts : list[str], categories: list[list[str]]):
        pk_field = "doc_id"
        dense_field = "dense_vector"
        sparse_field = "sparse_vector"
        text_field = "text"
        category_field = "category"
        dense_dim = 1024
        entities = []
        #reinit corpus in sparse embedding
        self.sparse_embedding_func = self.__init_sparse_embedding(texts)
        
        ## update old sparse embedding
        corpus = [(entity['doc_id'],entity['text'], entity[dense_field],entity[text_field], entity[category_field]) for entity in self.collection.query(
            expr="doc_id != ''", 
            output_fields=["doc_id","text", dense_field,text_field,category_field],
            consistency_level="Strong"
        )]

        if len(corpus) > 0: 
            upsert_entities = []       
            for doc_id,text,dense_field_value,text_field_value, category_field_value in corpus:
                entity = {
                    pk_field: doc_id,
                    sparse_field: self.sparse_embedding_func.embed_documents([text])[0],
                    dense_field : dense_field_value,
                    text_field : text_field_value,
                    category_field : category_field_value
                }
                upsert_entities.append(entity)
            self.collection.upsert(upsert_entities)

        ## insert new entities
        new_ids = [get_next_id() for _ in range(len(texts))]
        for i,(text,category) in enumerate(zip(texts, categories)):
            entity = {
                pk_field: new_ids[i] ,
                dense_field: self.dense_embedding_func.embed_documents([text])[0],
                sparse_field: self.sparse_embedding_func.embed_documents([text])[0],
                text_field: text,
                category_field: category,
            }
            entities.append(entity)
        res = self.collection.insert(entities)
        
        return res, new_ids
    
    def upsert(self,ids : list[str],texts : list[str],categories: list[list[str]]):
        """
        1.Remove old entities
        2.Update sparse embedding
        3.Update old sparse embeddings
        4.Insert new entities
        """
        pk_field = "doc_id"
        dense_field = "dense_vector"
        sparse_field = "sparse_vector"
        text_field = "text"
        category_field = "category"
        dense_dim = 1024
        entities = []
        self.remove(ids)
        
        #reinit corpus in sparse embedding
        self.sparse_embedding_func = self.__init_sparse_embedding(texts)
        
        ## update old sparse embedding
        corpus = [(entity['doc_id'],entity['text'], entity[dense_field],entity[text_field], entity[category_field]) for entity in self.collection.query(
            expr="doc_id != ''", 
            output_fields=["doc_id","text", dense_field,text_field,category_field],
            consistency_level="Strong"
        )]

        if len(corpus) > 0: 
            upsert_entities = []       
            for doc_id,text,dense_field_value,text_field_value, category_field_value in corpus:
                entity = {
                    pk_field: doc_id,
                    sparse_field: self.sparse_embedding_func.embed_documents([text])[0],
                    dense_field : dense_field_value,
                    text_field : text_field_value,
                    category_field : category_field_value
                }
                upsert_entities.append(entity)
            self.collection.upsert(upsert_entities)

        ## insert updated entities
        for i,(text,category) in enumerate(zip(texts, categories)):
            entity = {
                pk_field: ids[i] ,
                dense_field: self.dense_embedding_func.embed_documents([text])[0],
                sparse_field: self.sparse_embedding_func.embed_documents([text])[0],
                text_field: text,
                category_field: category,
            }
            entities.append(entity)
        res = self.collection.insert(entities)
        
        return res
    
    def remove(self, doc_ids : list[str]):
        return self.collection.delete(expr = f"doc_id in {doc_ids}", consistency_level="Strong")
    
    def get_articles(self, ids, categories):
        if len(ids) == 0 and len(categories) == 0:
            return self.collection.query(
            expr=f"doc_id != ''", 
            output_fields=["doc_id","text","category"],
            consistency_level="Strong"
        )
    
        return self.collection.query(
            expr=f"doc_id in {ids} and ARRAY_CONTAINS_ALL(category, {categories})",
            output_fields=["doc_id","text","category"],
            consistency_level="Strong"
        )

    def embed(self, text : str):
        return self.sparse_embedding_func.embed_query(text)


# texts = [
#     # ("In 'Whispers of the Forgotten' by Benjamin Hartley, an archaeologist named Alex delves into the secrets of an ancient temple, unearthing a sinister plot that spans centuries and threatens to unleash an unspeakable evil upon the world.", "Horror/Adventure"),
#     # ("In 'The Shattered Sky' by Isabella Nightingale, a group of space explorers embarks on a perilous journey to the edge of the universe, where they encounter cosmic horrors beyond imagination.", "Science Fiction/Horror"),
#     # ("In 'The Memory Eater' by Nathan Cross, a grieving widow discovers a device that allows her to relive cherished memories of her deceased husband, but soon realizes that altering the past comes with dire consequences.", "Thriller/Science Fiction"),
#     # ("In 'Echoes of the Deep' by Harper Rivers, a marine biologist uncovers a hidden underwater civilization threatened by human greed, forcing her to confront ethical dilemmas and the fragility of our oceans.", "Fantasy/Environmental"),
#     # ("In 'The Silent Symphony' by Gabriel Stone, a mute musician discovers that her compositions have the power to alter reality, leading her on a quest to harness her gift and save the world from an ancient evil.", "Fantasy/Magical Realism"),
#     # ("In 'The Clockwork Conspiracy' by Oliver Sparks, a detective in a steampunk city investigates a series of mysterious disappearances linked to a secret society plotting to overthrow the government with clockwork automatons.", "Steampunk/Mystery"),
#     # ("In 'The Forgotten Forest' by Willow Greene, a young girl with the ability to communicate with nature embarks on a quest to save her enchanted forest home from destruction by industrialists seeking to exploit its magic.", "Fantasy/Environmental"),
#     # ("In 'The Celestial Chronicles' by Luna Silverwood, a cosmic librarian must traverse the astral plane to retrieve stolen knowledge before it falls into the hands of dark forces intent on unraveling the fabric of reality.", "Fantasy/Adventure"),
#     # ("In 'The Alchemist's Apprentice' by Rowan Everhart, a novice alchemist discovers a forbidden formula that grants her incredible powers, but also attracts the attention of sinister forces eager to exploit her newfound abilities.", "Fantasy/Alchemy"),
#     # ("In 'The Dreamweaver's Dilemma' by Sage Moon, a young apprentice of dreams must confront her inner demons as she navigates a surreal dreamscape to rescue a lost soul trapped in the realm of nightmares.", "Fantasy/Adventure"),
#     # ("hello beast confront her inner demons as she navigates a surreal dreamscape to rescue a lost soul trapped in the realm of nightmares.", "Fantasy/Adventure")
#     # ("Whispers of the Forgotten' by Benjamin Hartley, an archaeologist named Alex delve, he started to surpass something because I do not know what to write, but it seemed fine for me", "Fantasy/Adventure")
#     ("In 'Whispers of the Forgotten' by Benjamin Hartley, an archaeologist named Alex delves into the secrets of an ancient temple, unearthing a sinister plot that spans centuries and threatens to unleash an unspeakable evil upon the world.", "Horror/Adventure"),
#     ("In 'Whispers of the Forgotten' by Benjamin Hartley, an archaeologist named Alex delves into the secrets of an ancient temple, unearthing a sinister plot that spans centuries and threatens to unleash an unspeakable evil upon the world.", "Horror/Adventure"),
#     ("In 'Whispers of the Forgotten' by Benjamin Hartley, an archaeologist named Alex delves into the secrets of an ancient temple, unearthing a sinister plot that spans centuries and threatens to unleash an unspeakable evil upon the world.", "Horror/Adventure"),
#     ("In 'Whispers of the Forgotten' by Benjamin Hartley, an archaeologist named Alex delves into the secrets of an ancient temple, unearthing a sinister plot that spans centuries and threatens to unleash an unspeakable evil upon the world.", "Horror/Adventure"),
#     ("In 'Whispers of the Forgotten' by Benjamin Hartley, an archaeologist named Alex delves into the secrets of an ancient temple, unearthing a sinister plot that spans centuries and threatens to unleash an unspeakable evil upon the world.", "Horror/Adventure"),
#     ("In 'Whispers of the Forgotten' by Benjamin Hartley, an archaeologist named Alex delves into the secrets of an ancient temple, unearthing a sinister plot that spans centuries and threatens to unleash an unspeakable evil upon the world.", "Horror/Adventure"),
#     ("In 'Whispers of the Forgotten' by Benjamin Hartley, an archaeologist named Alex delves into the secrets of an ancient temple, unearthing a sinister plot that spans centuries and threatens to unleash an unspeakable evil upon the world.", "Horror/Adventure"),
#     ("In 'Whispers of the Forgotten' by Benjamin Hartley, an archaeologist named Alex delves into the secrets of an ancient temple, unearthing a sinister plot that spans centuries and threatens to unleash an unspeakable evil upon the world.", "Horror/Adventure"),
#     ("In 'Whispers of the Forgotten' by Benjamin Hartley, an archaeologist named Alex delves into the secrets of an ancient temple, unearthing a sinister plot that spans centuries and threatens to unleash an unspeakable evil upon the world.", "Horror/Adventure"),
#     ("In 'Whispers of the Forgotten' by Benjamin Hartley, an archaeologist named Alex delves into the secrets of an ancient temple, unearthing a sinister plot that spans centuries and threatens to unleash an unspeakable evil upon the world.", "Horror/Adventure"),
#     ("In 'Whispers of the Forgotten' by Benjamin Hartley, an archaeologist named Alex delves into the secrets of an ancient temple, unearthing a sinister plot that spans centuries and threatens to unleash an unspeakable evil upon the world.", "Horror/Adventure"),
#     ("In 'Whispers of the Forgotten' by Benjamin Hartley, an archaeologist named Alex delves into the secrets of an ancient temple, unearthing a sinister plot that spans centuries and threatens to unleash an unspeakable evil upon the world.", "Horror/Adventure"),

# ]


# db = VectorDatabase()

# db.insert([text for text,category in texts], [category for text,category in texts])



