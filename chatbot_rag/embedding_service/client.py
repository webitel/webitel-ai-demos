import requests

english_text = "Hello world!"

response = requests.post("http://127.0.0.1:8000/embeddings", json={"text": "Example English text to embed."})
print(response)
response_body = response.json()
print(response_body['embedding'])