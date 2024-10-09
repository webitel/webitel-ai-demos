from flask import Flask, request, jsonify, Response
from src.tts import TTS_Module

app = Flask(__name__)

tts = TTS_Module()  # TTS_Module() #TTS_ElevenLabs()


@app.route("/tts", methods=["POST"])
def process_audio():
    if "text" not in request.json:
        return jsonify({"error": "No text provided"}), 400

    text = request.json["text"]
    stream = request.json.get("stream", False)

    if not stream:
        response_audio_path = tts.synthesize(text)

    def generate_audio_stream():
        if not stream:
            with open(response_audio_path, "rb") as f:
                while chunk := f.read(4096):
                    yield chunk
        else:
            for chunk in tts.synthesize_stream(text):
                yield chunk

    return Response(generate_audio_stream(), mimetype="audio/wav")


if __name__ == "__main__":
    app.run(debug=True, port=6000, host="0.0.0.0")
