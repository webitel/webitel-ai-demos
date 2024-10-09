from fastapi import FastAPI, Request
from transformers import AutoModel
import torch
import os

# Initialize the FastAPI app
app = FastAPI()

# Define the device for Torch (CPU in this case)
device = os.getenv("DEVICE")

# Initialize the HuggingFace embeddings model
model = (
    AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v3", trust_remote_code=True, torch_dtype=torch.float16
    )
    .to("cuda")
    .half()
)


# Endpoint to handle POST requests
@app.post("/embeddings/")
async def get_embeddings(request: Request):
    data = await request.json()
    text = data["text"]
    task = data.get("task", None)
    embedding = model.encode(text, task=task).tolist()
    return {"embedding": embedding}
