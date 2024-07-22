from fastapi import FastAPI, HTTPException
from extractor import (
    TopicExtractionRequest,
    TopicExtractionResponse,
    extract_topics_llm,
)

app = FastAPI()


@app.post("/extract_topics/", response_model=TopicExtractionResponse)
async def extract_topics(request: TopicExtractionRequest):
    try:
        chat_history_str = ""
        for message in request.messages:
            print(request.messages)
            chat_history_str += (
                message.sender.capitalize() + ": " + message.message + "\n"
            )
        response = extract_topics_llm(request.possible_topics, chat_history_str)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to extract topics: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=3333)
