import json
from typing import Optional


class StateManager:
    def __init__(self, json_path: str = "status.json"):
        self.json_path = json_path

    def load_last_date(self) -> Optional[int]:
        try:
            with open(self.json_path) as file:
                status = json.load(file)
                return status["last_date"]
        except FileNotFoundError:
            return None

    def save_last_date(self, last_date: int) -> None:
        with open(self.json_path, "w") as file:
            json.dump({"last_date": last_date}, file, indent=4)
