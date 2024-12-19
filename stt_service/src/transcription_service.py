import logging
import requests
from typing import Dict, Any, Tuple
from src.config import Config
from src.audio_processor import AudioProcessor


class TranscriptionService:
    def __init__(
        self, config: Config, webitel_conn, stt_model, audio_processor: AudioProcessor
    ):
        self.config = config
        self.webitel = webitel_conn
        self.stt = stt_model
        self.audio_processor = audio_processor
        self.setup_logging()

    def setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

    def enhance_text_with_punctuation(
        self, transcription: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.config.get("punctuation_api"):
            return transcription

        try:
            enhanced_chunks = []
            full_text = ""

            for chunk in transcription[0]["offsets"]:
                try:
                    response = requests.post(
                        self.config.get("punctuation_api"),
                        json={"text": chunk["text"]},
                        headers={"Content-Type": "application/json"},
                    )
                    response.raise_for_status()
                    enhanced_text = response.json()["enhanced_text"]
                    chunk["text"] = enhanced_text
                    full_text += enhanced_text + " "
                    enhanced_chunks.append(chunk)
                except requests.exceptions.RequestException as e:
                    self.logger.error(f"Punctuation API error: {e}")
                    full_text += chunk["text"] + " "
                    enhanced_chunks.append(chunk)

            transcription[0]["text"] = full_text.strip()
            transcription[0]["offsets"] = enhanced_chunks
            return transcription

        except Exception as e:
            self.logger.error(f"Error enhancing text: {e}")
            return transcription

    def process_audio_channel(self, audio_path: str) -> Tuple[Dict[str, Any], str]:
        transcription, language = self.stt.transcribe(audio_path, language=None)
        return self.enhance_text_with_punctuation(transcription), language

    def process_recording_and_upload(
        self, call_id: str, file_id: str, last_date: int
    ) -> None:
        try:
            # Download and process audio
            audio_path = self.webitel.download_audio(file_id)
            left_path, right_path = self.audio_processor.split_stereo_to_mono(
                audio_path
            )

            # Process each channel
            transcription_results = {}
            for channel, mono_path in enumerate([left_path, right_path]):
                result, language = self.process_audio_channel(mono_path)
                transcription_results[channel] = result

            # Upload results
            self.webitel.upload_transcription(
                call_id, file_id, transcription_results, language
            )

        except Exception as e:
            self.logger.error(f"Error processing recording {file_id}: {e}")
            raise

        finally:
            self.audio_processor.cleanup_files(audio_path, left_path, right_path)
