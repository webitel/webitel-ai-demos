from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
import io
import requests
import os
from typing import Optional
import weaviate
import base64
from base64 import b64encode
import json
import ast
from enums import IVRChoice
from src.simple_ordering_bot import SimpleOrderingBot
from src.templates import SimpleOrder2
import time
from typing import List, Union

app = FastAPI()

stt_url = os.getenv("STT_SERVICE_URL", "http://stt-service:5000/transcribe")
tts_url = os.getenv("TTS_SERVICE_URL", "http://tts-service:6000/tts")
weaviate_url = f"http://{os.getenv("HOST","weaviate"):{os.getenv("PORT","9999")}}"
streaming = False

bot = SimpleOrderingBot(SimpleOrder2)

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


mock_previous_order = {
    "name": "Ivan",
    "date": "дванадтяного вересня",
    "quantity": "чотири",
    "price": "456 гривень 50 копійок",
}


@app.post("/process-audio")
async def process_audio(
    file: Optional[UploadFile] = File(...),
    chat_history: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
    streaming: bool = Form(False),
    addresses: Optional[str] = Form(None),
    choice: str = Form(None),
    message: Optional[str] = Form(None),
):
    try:
        # Parse chat_history and context from headers if provided
        parsed_chat_history = []
        parsed_context = {}

        if chat_history:
            base64_decode = base64.b64decode(chat_history).decode("utf-8")
            print(base64_decode)
            parsed_chat_history = json.loads(base64_decode)
            parsed_chat_history = [
                ChatHistoryItem(**item) for item in parsed_chat_history
            ]

        if addresses:
            base64_decode = base64.b64decode(addresses).decode("utf-8")
            addresses = ast.literal_eval(f"{base64_decode}")

        if context:
            context = base64.b64decode(context).decode("utf-8")
            parsed_context = json.loads(context)

        # You can now use parsed_chat_history and parsed_context as needed
        # Example: print them for debugging purposes
        # print("Chat History:", parsed_chat_history)
        # print("Context:", parsed_context)

        # # Process the file and other form data here
        # # Example: print the streaming value and message_request
        # print("Streaming:", streaming)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400, detail="Invalid chat history or context JSON"
        )

    if choice:
        choice = int(choice)
    else:
        choice = -1

    answer = None

    if choice == IVRChoice.new_order.value:
        pass
    elif choice == IVRChoice.repeat_previous_order.value:
        transcribed_text = "Яке моє минуле замовлення?"
        answer = "На жаль, ви ще не робили замовлень у нас."
    elif choice == IVRChoice.discount.value:
        transcribed_text = "Які у вас акційні пропозиції?"
        answer = "Зараз діє акція на наступні товари : Чай Lovare Golden Ceylon за 101 гривню 50 копійок та Чай Ловар Багамський Саусеп за 139 гривень 50.  Бажаєте прослухати ще раз чи повернутися в головне меню?"
    elif choice == IVRChoice.connect_to_operator.value:
        transcribed_text = "з'єднати з оператором"
        answer = "Секундочку, з'єдную з оператором"
    # when button was already pressed
    else:
        if not message:
            # Read the audio file into memory
            audio_content = await file.read()
            # save file to mp4
            with open("audio_test.wav", "wb") as f:
                f.write(audio_content)

            # Send the audio file to the STT service
            try:
                stt_response = requests.post(
                    stt_url,
                    files={
                        "audio": ("audio.wav", io.BytesIO(audio_content), "audio/wav")
                    },
                )
            except Exception:
                # if we could not recognize the speech, we will return None
                return None
            if stt_response.status_code != 200:
                raise HTTPException(
                    status_code=stt_response.status_code, detail="STT service error"
                )

            transcription_result = stt_response.json()
            transcribed_text = transcription_result.get("transcription")
        else:
            transcribed_text = message
        print("Transcribed text:", transcribed_text)

        chat_history = []
        if parsed_chat_history:
            for message in parsed_chat_history:
                chat_history.append([message.message, message.sender])
        print("Chat History:", chat_history)
        print("Context:", parsed_context)

        if not transcribed_text:
            return None
            # raise HTTPException(status_code=500, detail="Failed to transcribe audio")

        try:
            bot_start = time.time()
            answer, context = bot.answer(
                transcribed_text, client, chat_history, parsed_context, addresses
            )
            bot_end = time.time()
            print("Bot time: ", bot_end - bot_start)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"BOT service error : {str(e)}")

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


# Define a type for chat history items

ChatHistoryItem2 = List[Union[str, str]]


class InputModel(BaseModel):
    user_input: str
    chat_history: Optional[List[ChatHistoryItem2]] = None
    context: Optional[dict] = None


class ResponseModel(BaseModel):
    response: str
    context: Optional[dict]
    end_conversation: bool


@app.post("/process-text", response_model=ResponseModel)
async def process_text(input_data: InputModel):
    user_input = input_data.user_input
    chat_history = input_data.chat_history
    context = input_data.context
    addresses = ["Адреса 1", "Адреса 2"]
    try:
        response, context = bot.answer(
            user_input, client, chat_history, context, addresses
        )
        end_conversation = context.pop("end_conversation")
        return ResponseModel(
            response=response, context=context, end_conversation=end_conversation
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing input: {str(e)}")
