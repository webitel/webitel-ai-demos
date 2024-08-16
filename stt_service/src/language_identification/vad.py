from silero_vad import load_silero_vad, read_audio, get_speech_timestamps


class VoiceDetection:
    def __init__(self):
        self.model = load_silero_vad()

    def detect_speech(self, audio_path: str, sampling_rate=16_000) -> list:
        wav = read_audio(audio_path, sampling_rate)
        speech_timestamps = get_speech_timestamps(
            wav, self.model, min_speech_duration_ms=100, min_silence_duration_ms=10000
        )
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
        # timestamps = self.merge_intervals(timestamps,2)
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

        return closest_timestamp, timestamps

    def merge_intervals(self, intervals, merge_t, buffer_time=0.2):
        # Sort intervals based on the start time
        intervals.sort(key=lambda x: x["start"])

        merged = []
        current_interval = intervals[0]

        for next_interval in intervals[1:]:
            # Check if we need to merge the current and next intervals
            if next_interval["start"] - current_interval["end"] < merge_t:
                # Merge intervals
                current_interval["end"] = max(
                    current_interval["end"], next_interval["end"]
                )
            else:
                # Add buffer time and update the merged interval
                merged_interval = {
                    "start": max(
                        current_interval["start"] - buffer_time, 0
                    ),  # Prevent negative start time
                    "end": current_interval["end"] + buffer_time,
                }
                merged.append(merged_interval)
                # Start a new interval
                current_interval = next_interval

        # Add the last interval with buffer time
        merged_interval = {
            "start": max(
                current_interval["start"] - buffer_time, 0
            ),  # Prevent negative start time
            "end": current_interval["end"] + buffer_time,
        }
        merged.append(merged_interval)

        return merged
