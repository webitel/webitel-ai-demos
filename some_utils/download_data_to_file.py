import weaviate

with weaviate.connect_to_local(host="localhost", port="9000") as client:
    collection = client.collections.get("KnowledgeBase")
    response = collection.query.fetch_objects(limit=25000, include_vector=True)

data = {}
data["objects"] = []
for obj in response.objects:
    content = obj.properties["content"]
    categories = obj.properties["categories"]
    vector = obj.vector["default"]
    data["objects"].append(
        {"object": {"content": content, "categories": categories}, "vector": vector}
    )

print(len(data["objects"]))
# json.dump(data,open("db_objects.json",'w'))
