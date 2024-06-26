import functools
import json
import os
import requests

import weaviate
from langchain_community.embeddings import HuggingFaceEmbeddings
from weaviate.classes.query import Filter

openai_model = os.environ.get("OPENAI_MODEL")
device = os.environ.get("DEVICE")
db_host = os.environ.get("HOST")
http_port = int(os.environ.get("HTTP_PORT"))
grpc_port = int(os.environ.get("GRPC_PORT"))


class VectorDatabase:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        # self.dense_embedding_func = HuggingFaceEmbeddings(
        #     model_name="ai-forever/sbert_large_nlu_ru", model_kwargs={"device": device}
        # )
        with weaviate.connect_to_local(host=host, port=port) as client:
            while not client.is_ready():
                pass
            if not client.collections.exists("KnowledgeBase"):
                schema = json.load(open("vector_db.json"))
                client.collections.create_from_dict(schema)

    def insert(
        self, contents: list[str], categories: list[list[str]]
    ) -> tuple[str, list[str]]:
        created_ids = []
        with weaviate.connect_to_local(host=self.host, port=self.port) as client:
            collection = client.collections.get("KnowledgeBase")
            with collection.batch.dynamic() as batch:
                for i, (content, category) in enumerate(zip(contents, categories)):
                    properties = {"content": content, "categories": category}
                    vector = self.get_dense_embedding(content)
                    res = batch.add_object(properties, vector=vector)
                    created_ids.append(str(res))

        if len(created_ids) == len(contents):
            return "All articles inserted successfully", created_ids

    def get_articles(self, ids, categories):
        with weaviate.connect_to_local(host=self.host, port=self.port) as client:
            collection = client.collections.get("KnowledgeBase")
            if len(ids) == 0 and len(categories) == 0:
                response = collection.query.fetch_objects()
            else:
                response = collection.query.fetch_objects(
                    limit=200,
                    filters=(
                        Filter.all_of(
                            [
                                Filter.by_id().contains_any(ids),
                                Filter.by_property("categories").contains_all(
                                    categories
                                ),
                            ]
                        )
                    ),
                )
        articles = []
        for object in response.objects:
            articles.append(
                {
                    "id": str(object.uuid),
                    "content": object.properties["content"],
                    "categories": object.properties["categories"],
                }
            )

        return articles

    def remove(self, ids) -> str:
        with weaviate.connect_to_local(host=self.host, port=self.port) as client:
            collection = client.collections.get("KnowledgeBase")
            delete_res = collection.data.delete_many(
                where=Filter.by_id().contains_any(ids)
            )
            if delete_res.successful == len(ids):
                return "All articles removed successfully"
            else:
                return "Some articles were not deleted"

    def upsert(self, ids, contents, categories):
        with weaviate.connect_to_local(host=self.host, port=self.port) as client:
            collection = client.collections.get("KnowledgeBase")
            for i, (id, content, category) in enumerate(zip(ids, contents, categories)):
                collection.data.update(
                    uuid=id,
                    properties={"content": content, "categories": category},
                    vector=self.get_dense_embedding(content),
                )
        return "All articles updated successfully"

    @functools.lru_cache(maxsize=128)
    def get_dense_embedding(self, text):
        response = requests.post("http://embedding_service:8000/embeddings", json={"text": text})
        return response.json()["embedding"]


if __name__ == "__main__":
    db = VectorDatabase(host="localhost", port=9000)
    db.insert(["Hello"], [["some category"]])
    res = db.get_articles(ids=[], categories=[])
    print(res)
