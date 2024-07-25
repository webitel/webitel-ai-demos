from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
from collections import defaultdict

# Import custom functions for training and inference
from model import train_model, inference

# Initialize FastAPI app
app = FastAPI()


# Define request models using Pydantic
class TrainRequest(BaseModel):
    dataset: Dict[str, List[str]] = Field(
        ...,
        description="A dictionary where the key is 'text' for a list of text samples and 'label' for a list of corresponding labels.",
        example={
            "text": [
                "I love this product!",
                "This is the worst experience I have ever had.",
                "Absolutely fantastic!",
                "Not good, not bad, just okay.",
                "Terrible, would not recommend.",
                "Great service and friendly staff.",
            ],
            "label": [2, 0, 2, 1, 0, 2],
        },
    )
    model_name: str = Field(
        ...,
        description="The name of the model to be trained.",
        example="custom_sentiment_model",
    )


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        description="The text for which sentiment prediction is to be made.",
        example="I really enjoyed the service!",
    )
    model_name: str = Field(
        ...,
        description="The name of the model to be used for prediction.",
        example="custom_sentiment_model",
    )


class ChatHistoryRequest(BaseModel):
    chat_history: List[Dict[str, str]] = Field(
        ...,
        description="A list of chat history entries where each entry contains a sender and a message.",
        example=[
            {"sender": "Alice", "message": "I really enjoyed the service!"},
            {"sender": "Bob", "message": "I hate the waiting time."},
            {"sender": "Alice", "message": "It was okay, not great."},
            {"sender": "Bob", "message": "Service was mediocre."},
        ],
    )
    model_name: str = Field(
        ...,
        description="The name of the model to be used for sentiment analysis.",
        example="sentiment_model_v1",
    )


# Define response models using Pydantic
class SentimentResult(BaseModel):
    label: str
    score: float


class PredictResponse(BaseModel):
    text: str
    sentiment_results: List[SentimentResult]


class SentimentCount(BaseModel):
    neutral: int
    negative: int
    positive: int


class ChatHistoryResponse(BaseModel):
    sentiment_counts: Dict[str, SentimentCount]


class TrainResponse(BaseModel):
    message: str
    training_results: dict


# Endpoint for training
@app.post(
    "/train/",
    response_model=TrainResponse,
    summary="Train a sentiment analysis model",
    description="Trains a sentiment analysis model with the provided dataset and model name.",
)
def train(train_request: TrainRequest):
    try:
        train_results = train_model(train_request.dataset, train_request.model_name)
        return {
            "message": "Training completed successfully",
            "training_results": train_results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint for prediction
@app.post(
    "/predict/",
    response_model=PredictResponse,
    summary="Predict sentiment of a given text",
    description="Uses the specified model to predict the sentiment of the provided text. Returns the sentiment results including labels and scores.",
)
def predict(predict_request: PredictRequest):
    try:
        # Perform inference using the inference function
        results = inference(predict_request.text)
        return {"text": predict_request.text, "sentiment_results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint for prediction on chat history
@app.post(
    "/predict_chat/",
    response_model=ChatHistoryResponse,
    summary="Predict sentiment for a chat history",
    description="Processes each message in the chat history to predict sentiment. Returns the counts of neutral, negative, and positive sentiments per sender.",
)
def predict_chat(predict_request: ChatHistoryRequest):
    try:
        # Dictionary to store sentiment counts per sender
        sentiment_counts = defaultdict(
            lambda: {"neutral": 0, "negative": 0, "positive": 0}
        )

        # Process each message in the chat history
        for entry in predict_request.chat_history:
            sender = entry["sender"]
            message = entry["message"]
            sentiment_results = inference(message)
            # Find the sentiment with the highest score
            if sentiment_results:
                max_sentiment = max(sentiment_results, key=lambda x: x["score"])[
                    "label"
                ]
                sentiment_counts[sender][max_sentiment] += 1

        return {"sentiment_counts": dict(sentiment_counts)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run FastAPI server with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
