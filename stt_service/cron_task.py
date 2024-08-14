import configparser
import logging
import json
import os
from src.stt_model_interface import STT
from src.webitel_connector import WebitelConnection
from src.utils import split_stereo_to_mono
from src.language_identification.language_identification import LanguageIndentification

logger = logging.getLogger(__name__)
logging.basicConfig(filename="app.log", encoding="utf-8", level=logging.DEBUG)

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

    # start loading and transcribing data
    for call_id, id, new_last_date in zip(call_ids, file_ids, last_dates):
        audio_path = ""
        try:
            audio_path = webitel_connection.download_audio(id)
            left_mono_path, right_mono_path = split_stereo_to_mono(audio_path, tmp_dir)
            transcription_results = []
            for mono_path in [left_mono_path, right_mono_path]:
                # may be necessary for other models
                lang_code = lang_detector.detect_language(mono_path, language_list)

                logging.info("Detected Language Code: %s" % lang_code)
                res = stt.transcribe(mono_path, language=lang_code)
                transcription_results.append(res)

            logger.info(f"Uploading transcription for call_id = {call_id}, id = {id}")
            webitel_connection.upload_transcription(call_id, id, transcription_results)

        except Exception as e:
            logger.error(
                f"Transcription failed for call_id = {call_id}, id = {id} due to {str(e)}"
            )
        finally:
            # +1 to ensure that we do not start transcribing the same recording again
            save_last_date(new_last_date + 1)
            if os.path.exists(audio_path):
                os.remove(audio_path)


if __name__ == "__main__":
    # TODO add scheduling from config
    job()
#     # schedule = scheduler.Scheduler()
#     # schedule.cyclic(dt.timedelta(minutes=1), job)

#     # while True:
#     #     schedule.exec_jobs()
#     #     time.sleep(1)
