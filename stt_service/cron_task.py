import configparser
import logging
import json
import os
from src.stt_model_interface import STT
from src.webitel_connector import WebitelConnection
from src.utils import split_stereo_to_mono, update_chunks
from src.language_identification.language_identification import LanguageIndentification
import librosa
from tempfile import NamedTemporaryFile
import soundfile as sf

logger = logging.getLogger(__name__)
logging.basicConfig(filename="app.log", encoding="utf-8", level=logging.INFO)

config = configparser.RawConfigParser()
json_path = "status.json"
tmp_dir = "temp_data"
config.read("speech_to_text.cfg")
details_dict = dict(config.items("CONFIG"))
details_dict["last_date"] = int(details_dict["last_date"])
language_list = json.loads(details_dict["languages"])

lang_detector = LanguageIndentification(device="cuda")
webitel_connection = WebitelConnection(
    details_dict["access_token"],
    details_dict["last_date"],
    tmp_dir,
    base_url=details_dict["base_url"],
)


def save_last_date(last_date):
    with open(json_path, "w") as file:
        json.dump({"last_date": last_date}, file, indent=4)


def job():
    try:
        status = json.load(open(json_path))
        webitel_connection.last_date = status["last_date"]
        logger.info(f"Using last_date = {status['last_date']} from local json")
    except FileNotFoundError:
        logger.error(f"{json_path} not found")

    # we get file ids using access_tokens starting from last_date time
    call_ids, file_ids, last_dates = webitel_connection.get_file_ids()
    # after we got ids we have updated last_date in webitel_connection
    # store it to local json

    stt = STT(details_dict["model_type"], details_dict)
    timestamp_processing = False
    # start loading and transcribing data
    for call_id, id, new_last_date in zip(call_ids, file_ids, last_dates):
        try:
            audio_path = "/home/nazar/Demo_Chatbot_retrieval/webitel-ai-demos/stt_service/temp_data/output.wav"
            # audio_path = webitel_connection.download_audio(id)
            left_mono_path, right_mono_path = split_stereo_to_mono(audio_path, tmp_dir)
            transcription_results = {}
            for channel, mono_path in enumerate([left_mono_path, right_mono_path]):
                # may be necessary for other models
                lang_code, timestamps = lang_detector.detect_language(
                    mono_path, language_list
                )
                logging.info("Detected Language Code: %s" % lang_code)
                logging.info("Detected Timestamps: %s" % timestamps)
                channel_result = {}
                if timestamp_processing and timestamps:
                    logger.info("Processing Timestamps Separately")
                    for j, timestamp in enumerate(timestamps):
                        start = timestamp["start"]
                        end = timestamp["end"]
                        logger.info(f"Start: {start}, End: {end}")
                        # Load the audio file with librosa
                        y, sr = librosa.load(mono_path, sr=16_000)
                        # Calculate the sample indices for the start and end times
                        start_sample = int(start * sr)
                        end_sample = int(end * sr)
                        # Extract the segment of interest
                        y_segment = y[start_sample:end_sample]
                        # Save the audio segment to a temporary file
                        with NamedTemporaryFile(
                            suffix=".wav", delete=False
                        ) as temp_audio_file:
                            temp_audio_path = (
                                temp_audio_file.name
                            )  # f"{channel}_{j}.wav"
                            sf.write(temp_audio_path, y_segment, sr)
                        try:
                            # Detect language on the trimmed audio segment
                            res, used_language = stt.transcribe(
                                temp_audio_path, language=lang_code
                            )
                            channel_result["text"] = (
                                channel_result.get("text", "") + res["text"]
                            )
                            channel_result["chunks"] = channel_result.get(
                                "chunks", []
                            ) + update_chunks(res["chunks"], start)
                        finally:
                            # Ensure the temporary file is deleted after processing
                            os.remove(temp_audio_path)
                    transcription_results[channel] = channel_result
                else:
                    res, used_language = stt.transcribe(mono_path, language=lang_code)
                    transcription_results[channel] = res
            print(transcription_results)
            logger.info(
                f"Uploading transcription for call_id = {call_id}, id = {id}, locale = {used_language}"
            )
            webitel_connection.upload_transcription(
                call_id, id, transcription_results, used_language
            )

        except Exception as e:
            logger.error(
                f"Transcription failed for call_id = {call_id}, id = {id} due to {str(e)}"
            )
        finally:
            # +1 to ensure that we do not start transcribing the same recording again
            # save_last_date(new_last_date + 1)
            # if os.path.exists(audio_path):
            #     os.remove(audio_path)
            pass
        break


if __name__ == "__main__":
    # TODO add scheduling from config
    job()
#     # schedule = scheduler.Scheduler()
#     # schedule.cyclic(dt.timedelta(minutes=1), job)

#     # while True:
#     #     schedule.exec_jobs()
#     #     time.sleep(1)
