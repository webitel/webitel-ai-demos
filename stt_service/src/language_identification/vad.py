from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


class VoiceDetection:
    def __init__(self):
        self.model = load_silero_vad()

    def detect_speech(self, audio_path: str, sampling_rate=16_000) -> list:
        wav = read_audio(audio_path, sampling_rate)
        speech_timestamps = get_speech_timestamps(wav, self.model)
        speech_timestamps_seconds = []
        for i in speech_timestamps:
            speech_timestamps_seconds.append(
                {"start": i["start"] / 16_000, "end": i["end"] / 16_000}
            )
        return speech_timestamps_seconds

    def find_closest_timestamp(
        self, audio_path: str, sampling_rate=16_000, target_s=30, pad=True
    ):
        timestamps = self.detect_speech(audio_path, sampling_rate)
        closest_timestamp = None
        closest_difference = float("inf")
        max_end = 0

        for timestamp in timestamps:
            difference = abs((timestamp["end"] - timestamp["start"]) - target_s)
            max_end = max(max_end, timestamp["end"])
            if difference < closest_difference:
                closest_difference = difference
                closest_timestamp = timestamp

        if pad:
            duration = closest_timestamp["end"] - closest_timestamp["start"]
            if duration < target_s:
                padding = (target_s - duration) / 2
                if closest_timestamp["start"] - padding > 0:
                    closest_timestamp["start"] -= padding
                if closest_timestamp["end"] + padding < max_end:
                    closest_timestamp["end"] += padding

        return closest_timestamp
