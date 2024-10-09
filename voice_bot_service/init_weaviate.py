import functools
import json
import requests
import weaviate
from weaviate.classes.query import Filter


class VectorDatabase:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        with weaviate.connect_to_local(host=host, port=port, grpc_port=50051) as client:
            while not client.is_ready():
                print("Collection not created")
                pass
            if not client.collections.exists("ProductList"):
                schema = json.load(open("db_collection.json"))
                client.collections.create_from_dict(schema)
                print("Collection created")
            else:
                print("Collection already exists")

    def insert(
        self,
        contents: list[str],
        prices: list[str],
        collection_name="ProductList",
    ) -> tuple[str, list[str]]:
        created_ids = []
        with weaviate.connect_to_local(
            host=self.host, port=self.port, grpc_port=50051
        ) as client:
            collection = client.collections.get(collection_name)
            with collection.batch.dynamic() as batch:
                for i, (content, price) in enumerate(zip(contents, prices)):
                    properties = {"content": content, "price": price}
                    vector = self.get_dense_embedding(content)
                    res = batch.add_object(properties, vector=vector)
                    created_ids.append(str(res))

        if len(created_ids) == len(contents):
            return "All articles inserted successfully", created_ids

    def get_articles(self, ids, categories, collection_name="ProductList"):
        with weaviate.connect_to_local(
            host=self.host, port=self.port, grpc_port=50051
        ) as client:
            collection = client.collections.get(collection_name)
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
                    "price": object.properties["price"],
                }
            )

        return articles

    def remove(self, ids, collection_name) -> str:
        with weaviate.connect_to_local(
            host=self.host, port=self.port, grpc_port=50051
        ) as client:
            collection = client.collections.get(collection_name)
            delete_res = collection.data.delete_many(
                where=Filter.by_id().contains_any(ids)
            )
            if delete_res.successful == len(ids):
                return "All articles removed successfully"
            else:
                return "Some articles were not deleted"

    def upsert(self, ids, contents, categories, collection_name="ProductList"):
        with weaviate.connect_to_local(
            host=self.host, port=self.port, grpc_port=50051
        ) as client:
            collection = client.collections.get(collection_name)
            for i, (id, content, category) in enumerate(zip(ids, contents, categories)):
                collection.data.update(
                    uuid=id,
                    properties={"content": content, "categories": category},
                    vector=self.get_dense_embedding(content),
                )
        return "All articles updated successfully"

    @functools.lru_cache(maxsize=128)
    def get_dense_embedding(self, text):
        response = requests.post(
            "http://localhost:4040/embeddings",
            json={"text": text, "task": "text-matching"},
        )
        res = response.json()["embedding"]
        return res


if __name__ == "__main__":
    db = VectorDatabase("localhost", 9000)
    # # print(db.insert(["test"], [["test"]]))

    json_data = json.load(
        open("mocked_products.json")
    )  # [{"product_name": "Чай Lovare Цитрусова Меліса (24 пак)"}, {"product_name": "Чай Lovare Golden Ceylon (50 пак)"}]
    contents = [item["product_name"].lower() for item in json_data]
    prices = [item["product_price"] for item in json_data]
    db.insert(contents, prices=prices)
    print(db.get_articles([], []))
    # print(json_data)

    # addresses = [
    #     "Бульвар Шевченка, 5a",
    #     "Вулиця Леніна, 12 ",
    #     "Вулиця Петра Порошенка 14",
    #     "Вулиця Лесі Українки 4",
    #     "Вулиця Шевченка 32б",
    # ]
    # addresses = [address.lower() for address in addresses]
    # db.insert(
    #     addresses, categories=[["test"]] * len(addresses), collection_name="Addresses"
    # )
    # print(db.get_articles([], [], collection_name="Addresses"))
