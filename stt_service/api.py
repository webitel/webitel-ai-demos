from src.stt_models.whisper_interface import FasterWhisper
from flask import Flask, request, jsonify

# create api with this model
stt_model = FasterWhisper(model_path="medium")

app = Flask(__name__)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    audio_file_path = "received_audio.wav"
    audio_file.save(audio_file_path)

    transcription_result, language = stt_model.transcribe(audio_file_path)
    return jsonify(
        {"transcription": transcription_result[0]["text"], "language": language}
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
