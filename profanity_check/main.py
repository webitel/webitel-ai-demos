from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from rapidfuzz import process, fuzz

app = FastAPI()


class SearchRequest(BaseModel):
    words: List[str]
    text: str
    confidence: Optional[int] = 80


class SearchResult(BaseModel):
    trigger_word: str
    id: int
    word_from_list: str
    confidence: float


@app.post("/search", response_model=List[SearchResult])
async def search_words(request: SearchRequest):
    word_list = request.words
    text = request.text
    confidence_threshold = request.confidence
    print(confidence_threshold)
    results = []

    for word in word_list:
        matches = process.extract(
            word, text.split(), scorer=fuzz.partial_ratio, limit=None
        )
        for match in matches:
            if (
                match[1] >= confidence_threshold
            ):  # Use the confidence threshold from the request
                results.append(
                    SearchResult(
                        trigger_word=match[0],
                        id=text.split().index(match[0]) + 1,
                        word_from_list=word,
                        confidence=match[1],
                    )
                )

    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
