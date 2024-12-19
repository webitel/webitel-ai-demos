import os
import librosa
import soundfile as sf
from typing import Tuple


class AudioProcessor:
    def __init__(self, tmp_dir: str):
        self.tmp_dir = tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)

    def split_stereo_to_mono(self, audio_path: str) -> Tuple[str, str]:
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        left_path = os.path.join(self.tmp_dir, "left.wav")
        right_path = os.path.join(self.tmp_dir, "right.wav")

        sf.write(left_path, y[0], sr)
        sf.write(right_path, y[1], sr)

        return left_path, right_path

    def cleanup_files(self, *file_paths: str) -> None:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
