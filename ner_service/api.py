from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gliner import GLiNER

app = FastAPI()
model = GLiNER.from_pretrained("numind/NuZero_token").to("cuda").half()
model.eval()


class RequestModel(BaseModel):
    text: str
    labels: list[str]


class EntityResponse(BaseModel):
    text: str
    label: str


def merge_entities(entities, text):
    merged_entities = []
    current_entity = None
    for entity in entities:
        if current_entity is None:
            current_entity = entity
        elif entity["start"] == current_entity["end"]:
            current_entity["end"] = entity["end"]
        else:
            # Check if there is anything between the entities in the original text
            print(text[current_entity["end"] : entity["start"]].strip())
            if text[current_entity["end"] : entity["start"]].strip() == "":
                current_entity["end"] = entity["end"]
                current_entity["text"] = text[
                    current_entity["start"] : current_entity["end"]
                ]
            else:
                merged_entities.append(current_entity)
                current_entity = entity

    if current_entity is not None:
        merged_entities.append(current_entity)

    return merged_entities


@app.post("/predict_entities", response_model=list[EntityResponse])
async def predict_entities(request: RequestModel):
    try:
        entities = model.predict_entities(request.text, request.labels)
        merged_entities = merge_entities(entities, request.text)

        return [
            {"text": entity["text"], "label": entity["label"]}
            for entity in merged_entities
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
