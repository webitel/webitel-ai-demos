import requests

access_token = "ekxyxpikp7yjtfeih97ddiauyr"
base_url = "https://cloud.webitel.ua/api"

url = f"{base_url}/call_center/audit/rate"

payload = {
    "call_id": "cc56699a-b6d2-41ff-a003-cbf9fb635c6a",
    "form": {"id": "6", "name": "AI Аудит дзвінка"},
    "answers": [
        {"name": "Ні", "score": 0},
        {"name": "Ні", "score": 10},
        {"name": "Ні", "score": 0},
        {"name": "Ні", "score": 0},
    ],
    "comment": "ai_test",
}

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "x-webitel-access": access_token,
}
print(payload)
response = requests.post(url, headers=headers, json=payload)
print(response.text)
