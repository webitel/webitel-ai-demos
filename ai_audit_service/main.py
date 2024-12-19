from src.webitel_connector import WebitelConnection
from src.llm_auditor import DSPY_auditor
import configparser
import os
import json

# SETTINGS
config = configparser.RawConfigParser()
config.read("audit.cfg")
config_dict = dict(config.items("CONFIG"))

OPENAI_API_KEY = config_dict["openai_api_key"]
access_token = config_dict["access_token"]
base_url = config_dict["base_url"]
last_date = int(config_dict["last_date"])
variable_name = config_dict["variable_name"]

json_path = "status.json"


def save_date(date: int):
    with open(json_path, "w") as file:
        json.dump({"last_date": date}, file, indent=4)


webitel_connection = WebitelConnection(
    access_token, base_url=base_url, last_date=last_date
)
llm_auditor = DSPY_auditor(api_key=OPENAI_API_KEY)


if __name__ == "__main__":
    if os.path.exists(json_path):
        with open(json_path, "r") as file:
            # Update last date with the last date from the json file
            data = json.load(file)
            last_date = data["last_date"]
            webitel_connection.last_date = last_date

    call_ids, _, last_dates, transcription_ids, audit_ids = (
        webitel_connection.get_file_ids(
            has_transcript=True, contains_variable=variable_name, rated=False
        )
    )
    for call_id, last_date, trans_id, audit_id in zip(
        call_ids, last_dates, transcription_ids, audit_ids
    ):
        transcription = webitel_connection.get_transcription(trans_id)
        questions_with_options_and_scores, audit_name = (
            webitel_connection.get_audit_questions(audit_id)
        )

        question_with_options = [
            (x[0], tuple(option[0] for option in x[1]))
            for x in questions_with_options_and_scores
        ]
        result, reasoning = llm_auditor.audit(transcription, question_with_options)
        result_with_scores = []
        print(call_id, result)
        for question, options_scores in questions_with_options_and_scores:
            answer = result[question]

            # Find matching score for the answer
            score = None
            for option_name, option_score in options_scores:
                if option_name == answer:
                    score = option_score
                    print(option_name, score)
                    break

            answer_dict = {"name": answer}
            answer_dict["score"] = score

            result_with_scores.append(answer_dict)

        webitel_connection.post_audit_result(
            call_id, audit_id, audit_name, result_with_scores, comment=reasoning
        )
        # save_date(last_date)
