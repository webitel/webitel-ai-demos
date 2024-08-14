import os
import librosa
import soundfile as sf
from tempfile import NamedTemporaryFile
from src.language_identification.speech_brain_language_detection import (
    SpeechBrainLanguageIdentification,
)
from src.language_identification.vad import VoiceDetection


class LanguageIndentification:
    def __init__(self, model_type="speechbrain", device="cpu"):
        self.model_type = model_type
        self.vad = VoiceDetection()
        if model_type == "speechbrain":
            self.model = SpeechBrainLanguageIdentification(device=device)
        else:
            raise Exception("Model type not supported")

    def detect_language(self, audio: str, languages: list[str], use_vad=True) -> str:
        detected_lang = None

        if use_vad:
            # Find the closest timestamp to extract the most likely spoken segment
            closest_timestamp = self.vad.find_closest_timestamp(audio)
            # If no speech is detected, return None
            if len(closest_timestamp) == 0:
                return detected_lang

            start, end = closest_timestamp["start"], closest_timestamp["end"]

            # Load the audio file with librosa
            y, sr = librosa.load(audio, sr=None)

            # Calculate the sample indices for the start and end times
            start_sample = int(start * sr)
            end_sample = int(end * sr)

            # Extract the segment of interest
            y_segment = y[start_sample:end_sample]

            # Save the audio segment to a temporary file

            with NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                temp_audio_path = temp_audio_file.name
                sf.write(temp_audio_path, y_segment, sr)

            try:
                # Detect language on the trimmed audio segment
                detected_lang = self.model.detect_language(temp_audio_path, languages)
            finally:
                # Ensure the temporary file is deleted after processing
                os.remove(temp_audio_path)
        else:
            detected_lang = self.model.detect_language(audio, languages)

        return detected_lang
