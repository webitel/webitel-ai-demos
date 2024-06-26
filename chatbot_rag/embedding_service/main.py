# Description: This file contains the code for the embedding service. The service is deployed using Ray Serve.
# Currently this is not used due to the fact that it takes 2-3 times more memory than FastAPI.

import starlette.requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from ray import serve
import torch

device = 'cpu'

@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    num_replicas=1,
)
class Model:
    def __init__(self):
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="ai-forever/sbert_large_nlu_ru", model_kwargs={"device": device, 'model_kwargs':{'torch_dtype':torch.float16}}
        )

    async def __call__(self, request: starlette.requests.Request):
        text: str = await request.json()
        return {'embedding': self.embeddings_model.embed_query(text)}

app = Model.bind()