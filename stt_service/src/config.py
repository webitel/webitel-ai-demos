import configparser
import json
from typing import Dict, Any


class Config:
    def __init__(self, config_path: str = "speech_to_text.cfg"):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        config = configparser.RawConfigParser()
        config.read(config_path)
        details = dict(config.items("CONFIG"))

        # Convert types and set defaults
        details["last_date"] = int(details["last_date"])
        details["languages"] = json.loads(details["languages"])
        details["variable_name"] = (
            None if details["variable_name"] == "none" else details["variable_name"]
        )
        details["min_talk_duration"] = (
            None
            if details["min_talk_duration"] == "none"
            else details["min_talk_duration"]
        )
        details["punctuation_api"] = (
            None if details["punctuation_api"] == "none" else details["punctuation_api"]
        )

        return details

    def get(self, key: str) -> Any:
        return self.config.get(key)
