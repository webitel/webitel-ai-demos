from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
import io
import requests
import os
from typing import Optional
from src.simple_ordering_bot import SimpleOrderingBot
from src.templates import SimpleOrder
import weaviate
import base64
from base64 import b64encode
import json
import ast

app = FastAPI()

stt_url = os.getenv("STT_SERVICE_URL", "http://stt-service:5000/transcribe")
tts_url = os.getenv("TTS_SERVICE_URL", "http://tts-service:6000/tts")
weaviate_url = f"http://{os.getenv("HOST","weaviate"):{os.getenv("PORT","9999")}}"
streaming = False

bot = SimpleOrderingBot(SimpleOrder)

client = weaviate.WeaviateClient(
    connection_params=weaviate.connect.ConnectionParams(
        http=weaviate.connect.ProtocolParams(
            host=os.getenv("HOST", "weaviate"),
            port=int(os.getenv("HTTP_PORT", 8080)),
            secure=False,
        ),
        grpc=weaviate.connect.ProtocolParams(
            host="webitel-ai-demos-weaviate-1",
            port=int(os.getenv("GRPC_PORT", 50051)),
            secure=False,
        ),
    )
)
client.connect()

# mocked  data
# addresses = [
#     "Бульвар Шевченка, 5a",
#     "Вулиця Леніна, 12 ",
#     "Вулиця Петра Порошенка",
#     "Вулиця Лесі Українки 4",
#     "Вулиця Шевченка 32б",
# ]


# Define the schema for chat history items
class ChatHistoryItem(BaseModel):
    message: str
    sender: str


@app.post("/process-audio/")
async def process_audio(
    file: UploadFile = File(...),
    chat_history: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
    streaming: bool = Form(False),
    addresses: Optional[str] = Form(None),
):
    print("RECEIVED ADDRESS", addresses)
    try:
        # Parse chat_history and context from headers if provided
        parsed_chat_history = []
        parsed_context = {}

        if chat_history:
            print("Input Chat History:", chat_history)
            base64_decode = base64.b64decode(chat_history).decode("utf-8")
            print(base64_decode)
            parsed_chat_history = json.loads(base64_decode)
            print("kek", parsed_chat_history)
            parsed_chat_history = [
                ChatHistoryItem(**item) for item in parsed_chat_history
            ]
            print("lol", parsed_chat_history)

        if addresses:
            base64_decode = base64.b64decode(addresses).decode("utf-8")
            print("DECODED ADDRESS", base64_decode)
            addresses = ast.literal_eval(f"{base64_decode}")
            print("Addresses:", addresses, type(addresses))

        if context:
            print("Input Context:", context)
            context = base64.b64decode(context).decode("utf-8")
            parsed_context = json.loads(context)

        # You can now use parsed_chat_history and parsed_context as needed
        # Example: print them for debugging purposes
        print("Chat History:", parsed_chat_history)
        print("Context:", parsed_context)

        # Process the file and other form data here
        # Example: print the streaming value and message_request
        print("Streaming:", streaming)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Invalid chat history or context JSON"
        )

    # Read the audio file into memory
    audio_content = await file.read()

    # Send the audio file to the STT service
    stt_response = requests.post(
        stt_url, files={"audio": ("audio.wav", io.BytesIO(audio_content), "audio/wav")}
    )
    if stt_response.status_code != 200:
        raise HTTPException(
            status_code=stt_response.status_code, detail="STT service error"
        )

    transcription_result = stt_response.json()
    transcribed_text = transcription_result.get("transcription")

    chat_history = []
    if parsed_chat_history:
        for message in parsed_chat_history:
            chat_history.append([message.message, message.sender])
    print("Chat History:", chat_history)
    print("Context:", parsed_context)

    answer, context = bot.answer(
        transcribed_text, client, chat_history, parsed_context, addresses
    )

    if not transcribed_text:
        raise HTTPException(status_code=500, detail="Failed to transcribe audio")

    # Send the transcribed text to the TTS service
    tts_response = requests.post(tts_url, json={"text": answer, "stream": streaming})
    if tts_response.status_code != 200:
        raise HTTPException(
            status_code=tts_response.status_code, detail="TTS service error"
        )
    print("new context", context)
    if streaming:

        def generate_audio_stream():
            for chunk in tts_response.iter_content(chunk_size=4096):
                yield chunk

        return StreamingResponse(
            generate_audio_stream(),
            media_type="audio/wav",
            headers={
                "human": encode_answer(transcribed_text),
                "bot_answer": encode_answer(answer),
                "context": encode_answer(context),
            },
        )
    else:
        return StreamingResponse(
            io.BytesIO(tts_response.content),
            media_type="audio/wav",
            headers={
                "human": encode_answer(transcribed_text),
                "bot_answer": encode_answer(answer),
                "context": encode_answer(context),
            },
        )


def encode_answer(text):
    if isinstance(text, dict):
        return b64encode(json.dumps(text).encode("utf-8")).decode()
    return b64encode(text.encode("utf-8")).decode()
