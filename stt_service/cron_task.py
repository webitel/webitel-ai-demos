from src.config import Config
from src.state_manager import StateManager
from src.audio_processor import AudioProcessor
from src.transcription_service import TranscriptionService
from src.stt_model_interface import STT
from src.webitel_connector import WebitelConnection


def main():
    # Initialize components
    config = Config("speech_to_text.cfg")
    state_manager = StateManager("status.json")
    audio_processor = AudioProcessor("temp_data")

    webitel_connection = WebitelConnection(
        config.get("access_token"),
        state_manager.load_last_date() or config.get("last_date"),
        "temp_data",
        base_url=config.get("base_url"),
    )

    stt = STT(
        config.get("model_type"),
        config.config,
        config.get("device"),
        config.get("hf_token"),
    )

    service = TranscriptionService(config, webitel_connection, stt, audio_processor)

    # Process recordings
    call_ids, file_ids, last_dates, _, _ = webitel_connection.get_file_ids(
        contains_variable=config.get("variable_name"),
        min_talk_duration=config.get("min_talk_duration"),
    )

    for call_id, file_id, new_last_date in zip(call_ids, file_ids, last_dates):
        try:
            service.process_recording_and_upload(call_id, file_id, new_last_date)
        finally:
            state_manager.save_last_date(new_last_date + 1)


if __name__ == "__main__":
    main()
