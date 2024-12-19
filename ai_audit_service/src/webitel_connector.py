import requests
import time
import os


class WebitelConnection:
    """Class to download audiofiles and upload transcription"""

    def __init__(self, access_token, last_date=0, tmp_dir="temp_data", base_url=None):
        self.access_token = access_token
        self.last_date = last_date
        self.tmp_dir = tmp_dir
        if base_url is None:
            raise Exception("Base url is not provided")
        self.base_url = base_url
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)

    def download_audio(self, recording_id):
        """Downloads an audio file given an access token and recording ID."""
        url = f"{self.base_url}/storage/recordings/{recording_id}/download?access_token={self.access_token}"
        headers = {
            "Accept": "audio/mpeg",  # Adjust this if necessary based on the actual file type
        }

        response = requests.get(url, headers=headers, stream=True)

        if response.status_code == 200:
            filename = f"{self.tmp_dir}/recording_{recording_id}.mp3"  # Adjust file extension if necessary
            with open(filename, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            return filename
        else:
            raise Exception(
                f"Error occured while downloading data. Status code {response.status_code}, message {response.text}"
            )

    def upload_transcription(
        self, call_id, file_id, transcription_data: dict[dict], used_language
    ):
        """Uploads transcription data to the server."""
        url = f"{self.base_url}/storage/transcript_file"

        # Prepare data in the required format
        phrases = []
        text = ""
        for channel, transcription in transcription_data.items():
            for chunk in transcription[0]["offsets"]:
                phrases.append(
                    {
                        "channel": channel,
                        "start_sec": chunk["timestamp"][0],
                        "end_sec": chunk["timestamp"][1],
                        "phrase": chunk["text"],
                    }
                )
            text = f"Channel {channel} : {transcription[0]['text']} \n"

        payload = {
            "file_id": file_id,
            "locale": used_language,
            "phrases": phrases,
            "text": text,
            "uuid": call_id,
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-webitel-access": self.access_token,
        }

        response = requests.put(url, headers=headers, json=payload)

        if response.status_code == 200:
            print("Transcription uploaded successfully")
        else:
            raise Exception(
                f"Failed to upload transcription. Status code {response.status_code}, message {response.text}"
            )

    def get_file_ids(
        self,
        has_transcript=False,
        contains_variable=None,
        min_talk_duration=5,
        rated=None,
    ):
        file_ids = []
        transcriptions_ids = []
        audit_ids = []
        call_ids = []
        last_dates = []
        has_more_data = True
        last_date = self.last_date
        current_time = int(time.time() * 1000)
        while has_more_data:
            body = {
                "size": 10,
                "has_file": True,
                "skip_parent": True,  # leg A
                "stored_at": {"from": last_date, "to": current_time},
                "sort": "stored_at",
                "fields": ["id", "files", "stored_at", "transcripts", "variables"],
                "talk": {"from": str(min_talk_duration), "to": None},
                "has_transcript": has_transcript,
            }
            if rated is not None:
                body["rated"] = rated

            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "x-webitel-access": self.access_token,
            }
            response = requests.post(
                self.base_url + "/calls/history", headers=headers, json=body
            )
            if response.status_code == 200:
                list_data = response.json()
                if "items" in list_data:
                    for item in list_data["items"]:
                        # Need to increment last_date to last known processed date
                        last_date = int(item["stored_at"])
                        self.last_date = max(self.last_date, last_date)

                        if (
                            contains_variable is not None
                            and contains_variable
                            not in list(item.get("variables", {}).keys())
                        ):
                            print(
                                f"Skipping {item['id']} cause - does not contain variable {contains_variable}"
                            )
                            continue
                        elif (
                            contains_variable is not None
                            and contains_variable
                            in list(item.get("variables", {}).keys())
                        ):
                            audit_ids.append(
                                item.get("variables", {})[contains_variable]
                            )
                        if has_transcript:
                            # get only first

                            transcriptions_ids.append(item["transcripts"][0]["id"])

                        call_ids.append(item["id"])

                        # for now only first file is used
                        file_ids.append(item["files"][0]["id"])

                        last_dates.append(last_date)
                    # Check if there's more data to fetch
                    has_more_data = list_data.get("next", False)
                else:
                    has_more_data = False
            else:
                print(
                    f"Request failed with status code {response.status_code}, message {response.text}"
                )
                has_more_data = False

        return call_ids, file_ids, last_dates, transcriptions_ids, audit_ids

    def get_transcription(self, transcription_id):
        url = (
            self.base_url
            + f"/storage/transcript_file/{transcription_id}/phrases?page=0&size=100"
        )
        headers = {
            "accept": "application/json",
            "x-webitel-access": self.access_token,
        }

        response = requests.get(url, headers=headers)

        return response.json()

    def get_audit_questions(self, audit_id):
        url = self.base_url + f"/call_center/audit/forms/{audit_id}"
        headers = {
            "accept": "application/json",
            "x-webitel-access": self.access_token,
        }
        questions_with_options_and_scores = []
        response = requests.get(url, headers=headers)
        audit_name = None
        if response.status_code == 200:
            json_response = response.json()
            for audit_question in json_response["questions"]:
                question = audit_question["question"]
                options_and_scores = [
                    (option["name"], option.get("score", 0))
                    for option in audit_question["options"]
                ]
                questions_with_options_and_scores.append((question, options_and_scores))
            audit_name = json_response["name"]
        else:
            print(
                f"Could not get audit with id {audit_id}, error : {str(response.text)}"
            )

        return questions_with_options_and_scores, audit_name

    def post_audit_result(
        self,
        call_id: str,
        form_id: str,
        form_name: str,
        answers: list,
        comment: str = "",
    ):
        """Posts audit results to Webitel API.

        Args:
            call_id: ID of the call being audited
            form_id: ID of the audit form
            form_name: Name of the audit form
            answers: List of dictionaries containing scores for each question
            comment: Optional comment for the audit
        """
        url = f"{self.base_url}/call_center/audit/rate"

        payload = {
            "call_id": call_id,
            "form": {"id": form_id, "name": form_name},
            "answers": answers,
            "comment": comment,
        }

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-webitel-access": self.access_token,
        }
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            raise Exception(
                f"Failed to post audit results. Status code {response.status_code}, message {response.text}"
            )

        return response.json()
