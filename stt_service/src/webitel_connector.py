import requests
import time
import os
from collections import Counter


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

    def upload_transcription(self, call_id, file_id, transcription_data: list[dict]):
        """Uploads transcription data to the server."""
        url = f"{self.base_url}/storage/transcript_file"

        # Prepare data in the required format
        phrases = []
        text = ""
        locale_occurances = []
        for channel, transcription in enumerate(transcription_data):
            for chunk in transcription["chunks"]:
                phrases.append(
                    {
                        "channel": channel,
                        "start_sec": chunk["timestamp"][0],
                        "end_sec": chunk["timestamp"][1],
                        "phrase": chunk["text"],
                    }
                )
                locale_occurances.append(chunk["language"])
            text = f"Channel {channel} : {transcription['text']} \n"

        locale_counter = Counter(locale_occurances)

        payload = {
            "file_id": file_id,
            "locale": locale_counter.most_common(1)[0][0],
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

    def get_file_ids(self):
        file_ids = []
        call_ids = []
        last_dates = []
        has_more_data = True
        last_date = self.last_date

        while has_more_data:
            body = {
                "size": 100,
                "has_file": True,
                "skip_parent": True,  # leg A
                "stored_at": {"from": last_date, "to": int(time.time() * 1000)},
                "sort": "stored_at",
                "fields": ["id", "files", "stored_at"],
            }

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
                        call_ids.append(item["id"])
                        # for now only first file is used
                        file_ids.append(item["files"][0]["id"])
                        last_date = int(item["stored_at"])
                        self.last_date = max(self.last_date, last_date)
                        last_dates.append(last_date)
                    # Check if there's more data to fetch
                    has_more_data = False  # list_data.get('next', False)
                else:
                    has_more_data = False
            else:
                print(f"Request failed with status code {response.status_code}")
                has_more_data = False

        return call_ids, file_ids, last_dates
