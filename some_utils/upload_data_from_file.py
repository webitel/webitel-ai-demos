import weaviate
import json

db_data = json.load(open("db_objects.json"))

with weaviate.connect_to_local(host="localhost", port="9000") as client:
    collection = client.collections.get("KnowledgeBase")
    with collection.batch.dynamic() as batch:
        for i, data_object in enumerate(db_data["objects"]):
            properties = data_object["object"]
            vector = data_object["vector"]
            res = batch.add_object(properties, vector=vector)
