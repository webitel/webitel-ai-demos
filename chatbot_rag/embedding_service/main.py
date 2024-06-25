import starlette.requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from ray import serve
import ray


# ray.init(dashboard_host="0.0.0.0",dashboard_port=8265)
device = 'cpu'

@serve.deployment(
    ray_actor_options={"num_cpus": 1},
    num_replicas="auto",
)
class Model:
    def __init__(self):
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="ai-forever/sbert_large_nlu_ru", model_kwargs={"device": device}
        )

    async def __call__(self, request: starlette.requests.Request):
        text: str = await request.json()
        return {'embedding': self.embeddings_model.embed_query(text)}
#deploy
# ray.init(dashboard_host="0.0.0.0",dashboard_port=8265)
app = Model.bind()