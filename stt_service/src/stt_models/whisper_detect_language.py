from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
)
from datasets import Audio, Dataset


class LanguageDetector:
    """Whisper for language detection"""

    def __init__(self) -> None:
        model_path = "openai/whisper-medium"
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = WhisperTokenizer.from_pretrained(model_path)

    def detect_language(self, audio_path: str):
        # Load and resample the audio to 16 kHz using datasets.Audio
        audio_dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column(
            "audio", Audio(sampling_rate=16000)
        )
        audio = audio_dataset[0]["audio"]["array"]

        # Preprocess the audio input
        input_features = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features

        # Generate the language token
        lang_token = self.model.generate(input_features, max_new_tokens=1)[0, 1]

        # Decode the language token to get the language code
        language_code = self.tokenizer.decode(lang_token)

        # Print the detected language code
        print(language_code)
        return language_code


if __name__ == "__main__":
    lang_detector = LanguageDetector()
    lang_detector.detect_language(
        "/home/samael/md1/STT_YT_DATA/data/fffd413a75a4024a6162bc133bf12263.mp3"
    )
