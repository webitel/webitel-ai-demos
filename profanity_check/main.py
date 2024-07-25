from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from rapidfuzz import process, fuzz
import nltk
from nltk.util import ngrams
import re

# Download NLTK data
nltk.download("punkt")


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


def generate_ngrams(tokens, n):
    return [" ".join(ngram) for ngram in ngrams(tokens, n)]


@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    word_list = request.words
    text = request.text
    confidence_threshold = request.confidence
    results = []

    # Tokenize the text and remove punctuation
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [
        re.sub(r"\W+", "", token) for token in tokens if re.sub(r"\W+", "", token) != ""
    ]

    # Determine the maximum n-gram length
    max_ngram_length = max(len(word.split()) for word in word_list)

    # Generate n-grams for each length up to the maximum
    ngrams_dict = {
        n: generate_ngrams(filtered_tokens, n) for n in range(1, max_ngram_length + 1)
    }

    for word in word_list:
        word_length = len(word.split())
        ngram_strings = ngrams_dict.get(word_length, [])

        matches = process.extract(
            word, ngram_strings, scorer=fuzz.partial_ratio, limit=None
        )
        for match in matches:
            if match[1] >= confidence_threshold:
                results.append(
                    SearchResult(
                        trigger_word=match[0],
                        id=ngram_strings.index(match[0]) + 1,
                        word_from_list=word,
                        confidence=match[1],
                    )
                )

    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
