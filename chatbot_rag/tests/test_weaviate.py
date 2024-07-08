import weaviate


with weaviate.connect_to_local(host="localhost", port="9000") as client:
    collection = client.collections.get("KnowledgeBase")
    response = collection.query.fetch_objects(limit=25000)
print(response.objects)
print(len(response.objects))
