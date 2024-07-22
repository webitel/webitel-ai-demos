# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import custom functions for training and inference
from model import train_model, inference

# Initialize FastAPI app
app = FastAPI()


# Define request models using Pydantic
class TrainRequest(BaseModel):
    dataset: dict
    model_name: str


class PredictRequest(BaseModel):
    text: str
    model_name: str


# Endpoint for training
@app.post("/train/")
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
@app.post("/predict/")
def predict(predict_request: PredictRequest):
    try:
        # Perform inference using the inference function
        results = inference(predict_request.text)
        return {"text": predict_request.text, "sentiment_results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run FastAPI server with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
