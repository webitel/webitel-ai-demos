from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from nemo.collections.nlp.models import PunctuationCapitalizationModel
import time
import uvicorn

app = FastAPI(
    title="Ukrainian Punctuation and Capitalization API",
    description="API for restoring punctuation and capitalization in Ukrainian text",
    version="1.0.0",
)

# Config
MODEL_NAME = "dchaplinsky/punctuation_uk_bert"

# Load the model at startup
model = PunctuationCapitalizationModel.from_pretrained(MODEL_NAME)


class TextRequest(BaseModel):
    text: str


class TextResponse(BaseModel):
    enhanced_text: str
    elapsed_time: float


class BatchTextRequest(BaseModel):
    texts: List[str]


class BatchTextResponse(BaseModel):
    results: List[TextResponse]
    total_time: float


@app.post("/enhance", response_model=TextResponse)
async def enhance_text(request: TextRequest):
    """
    Enhance a single text with punctuation and capitalization.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    start_time = time.time()

    predictions = model.add_punctuation_capitalization([request.text.strip()])
    enhanced_text = predictions[0] if predictions else request.text

    elapsed_time = time.time() - start_time

    return TextResponse(
        enhanced_text=enhanced_text.strip(), elapsed_time=round(elapsed_time, 2)
    )


@app.post("/enhance/batch", response_model=BatchTextResponse)
async def enhance_batch(request: BatchTextRequest):
    """
    Enhance multiple texts with punctuation and capitalization.
    """
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")

    start_time = time.time()
    results = []

    for text in request.texts:
        if not text.strip():
            continue

        text_start_time = time.time()
        predictions = model.add_punctuation_capitalization([text.strip()])
        enhanced_text = predictions[0] if predictions else text
        text_elapsed_time = time.time() - text_start_time

        results.append(
            TextResponse(
                enhanced_text=enhanced_text.strip(),
                elapsed_time=round(text_elapsed_time, 2),
            )
        )

    total_time = time.time() - start_time

    return BatchTextResponse(results=results, total_time=round(total_time, 2))


# Example texts endpoint for testing
@app.get("/examples", response_model=List[str])
async def get_examples():
    """
    Get example texts for testing the API.
    """
    return [
        "тема про яку не люблять говорити офіційні джерела у генштабі і міноборони це хімічна зброя окупанти вже тривалий час використовують хімічну зброю заборонену",
        "всіма конвенціями якщо спочатку це були гранати з дронів то тепер фіксують випадки застосування",
        "хімічних снарядів причому склад отруйної речовони різний а отже й наслідки для наших військових теж різні",
    ]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
