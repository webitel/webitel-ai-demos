# !pip install git+https://github.com/robinhad/ukrainian-tts.git
from ukrainian_tts.tts import TTS, Voices, Stress
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import os
from os.path import join
from espnet2.bin.tts_inference import Text2Speech
import torch


ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")


class ImprovedTTS(TTS):
    def __init__(self, cache_folder=None, device="cpu", dtype="float16") -> None:
        """
        Class to set up a text-to-speech engine, from download to model creation.
        Downloads or uses files from `cache_folder` directory.
        By default, stores in the current directory.

        Args:
            cache_folder (str): Directory to cache model files.
            device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
            dtype (str): Data type for model parameters (e.g., 'float16' or 'float32').
        """
        self.cache_folder = cache_folder if cache_folder is not None else "."
        super().__init__(
            cache_folder, device
        )  # Initialize parent class with cache_folder and device
        self.reinit(dtype)  # Initialize synthesizer

    def reinit(self, dtype="float16"):
        """
        Reinitialize the TTS synthesizer with the specified model files.

        Args:
            dtype (str): Data type for model parameters (e.g., 'float16' or 'float32').
        """
        model_path = join(self.cache_folder, "model.pth")
        config_path = join(self.cache_folder, "config.yaml")
        speakers_path = join(self.cache_folder, "spk_xvector.ark")
        feat_stats_path = join(self.cache_folder, "feats_stats.npz")

        # Ensure the required files exist
        for path in [model_path, config_path, speakers_path, feat_stats_path]:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Required file '{path}' not found.")

        # Initialize the Text2Speech synthesizer in the parent class
        self.synthesizer = Text2Speech(
            train_config=config_path,
            model_file=model_path,
            device=self.device,
            dtype=dtype,
        )
        print("TTS synthesizer initialized successfully.")


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
        torch.cuda.empty_cache()
        return "test.wav"

    def synthetize_stream(self, text):
        return "test.wav"


class TTS_ElevenLabs:
    def __init__(self, api_key=ELEVENLABS_API_KEY):
        self.api_key = api_key
        self.client = ElevenLabs(
            api_key=api_key,
        )
        self.voice_id = "AAf5O13lYHNKE7VJQL4x"  # "pMsXgVXv3BLzUgSXRplE"

    def synthesize(self, text, **kwargs):
        return self.text_to_speech_file(text)

    def text_to_speech_file(self, text: str) -> str:
        # Calling the text_to_speech conversion API with detailed parameters
        response = self.client.text_to_speech.convert(
            voice_id=self.voice_id,  # Adam pre-made voice
            # output_format="ulaw_8000",
            text=text,
            # model_id="eleven_multilingual_v2",  # use the turbo model for low latency
            model_id="eleven_turbo_v2_5",  # use the turbo model for low latency
            # language_code="uk",
            voice_settings=VoiceSettings(
                stability=0.52,
                similarity_boost=0.69,
                style=0.35,
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
            # output_format="ulaw_8000",
            text=text,
            # model="eleven_multilingual_v2",  # use the turbo model for low latency
            model="eleven_turbo_v2_5",  # use the turbo model for low latency
            # language_code='uk',
            voice_settings=VoiceSettings(
                stability=0.52,
                similarity_boost=0.69,
                style=0.35,
                use_speaker_boost=True,
            ),
            stream=True,
        )
