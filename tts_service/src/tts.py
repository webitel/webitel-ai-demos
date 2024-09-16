# !pip install git+https://github.com/robinhad/ukrainian-tts.git
from ukrainian_tts.tts import TTS, Voices, Stress
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import os

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


class TTS_Module:
    def __init__(self, device="cuda"):
        self.device = device
        self.tts = TTS(device="cuda")

    def synthesize(
        self, text, voice=Voices.Tetiana.value, stress=Stress.Dictionary.value
    ):
        with open("test.wav", mode="wb") as file:
            _, output_text = self.tts.tts(text, voice, stress, file)
            print("Accented text:", output_text)
        return "test.wav"

    def synthetize_stream(self, text):
        return "test.wav"


class TTS_ElevenLabs:
    def __init__(self, api_key=ELEVENLABS_API_KEY):
        self.api_key = api_key
        self.client = ElevenLabs(
            api_key=api_key,
        )
        self.voice_id = "pMsXgVXv3BLzUgSXRplE"

    def synthesize(self, text, **kwargs):
        return self.text_to_speech_file(text)

    def text_to_speech_file(self, text: str) -> str:
        # Calling the text_to_speech conversion API with detailed parameters
        response = self.client.text_to_speech.convert(
            voice_id=self.voice_id,  # Adam pre-made voice
            # output_format="ulaw_8000",
            text=text,
            model_id="eleven_turbo_v2_5",  # use the turbo model for low latency
            language_code="uk",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )

        # Generating a unique file name for the output MP3 file
        save_file_path = "test.wav"

        # Writing the audio to a file
        with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

        print(f"{save_file_path}: A new audio file was saved successfully!")

        # Return the path of the saved audio file
        return save_file_path

    def synthesize_stream(self, text):
        return self.client.generate(
            voice=self.voice_id,  # Adam pre-made voice
            output_format="ulaw_8000",
            text=text,
            model="eleven_turbo_v2_5",  # use the turbo model for low latency
            # language_code='uk',
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=1.0,
                style=0.0,
                use_speaker_boost=True,
            ),
            stream=True,
        )
