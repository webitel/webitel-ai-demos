from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from src.stt_models.my_w2v_bert import W2V_BERT_WITH_LM
import time
import io

# Create an instance of the FastAPI app
app = FastAPI()

# Initialize the STT model
# stt_model = FasterWhisper(model_path="deepdml/faster-whisper-large-v3-turbo-ct2", batched=False)
# stt_w2v_bert = W2V_BERT()
stt_with_lm = W2V_BERT_WITH_LM()


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # Read audio file into memory
    audio_bytes = await file.read()
    audio_file_io = io.BytesIO(audio_bytes)

    # Measure transcription time
    start_trans = time.time()
    # transcription_result, language = stt_model.transcribe(audio_file_io, language="uk")[0]["text"]
    transcription_result, language = stt_with_lm.transcribe(audio_file_io)
    end_trans = time.time()

    # Print transcription time
    print("Trans time", end_trans - start_trans)

    return JSONResponse(
        content={
            "transcription": transcription_result,
            # "language": language
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
