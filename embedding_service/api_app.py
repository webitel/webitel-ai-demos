from fastapi import FastAPI, Request
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
import os

# Initialize the FastAPI app
app = FastAPI()

# Define the device for Torch (CPU in this case)
device = os.getenv("DEVICE")

# Initialize the HuggingFace embeddings model
embeddings_model = HuggingFaceEmbeddings(
    model_name="ukr-models/xlm-roberta-base-uk",
    model_kwargs={"device": device, "model_kwargs": {"torch_dtype": torch.float16}},
)


# Endpoint to handle POST requests
@app.post("/embeddings/")
async def get_embeddings(request: Request):
    data = await request.json()
    text = data["text"]
    embedding = embeddings_model.embed_query(text)
    return {"embedding": embedding}
