from pydub import AudioSegment
import os


def split_stereo_to_mono(stereo_file_path, tmp_dir):
    left_output_path, right_output_path = (
        os.path.join(tmp_dir, "left.wav"),
        os.path.join(tmp_dir, "right.wav"),
    )
    # Load the stereo audio file
    audio = AudioSegment.from_file(stereo_file_path)

    # Check if the audio is stereo
    if audio.channels != 2:
        raise ValueError("The provided audio file is not stereo")

    # Split stereo into left and right channels
    left_channel = audio.split_to_mono()[0]
    right_channel = audio.split_to_mono()[1]
    # Export the mono channels to separate files
    left_channel.export(left_output_path, format="wav")
    right_channel.export(right_output_path, format="wav")

    return left_output_path, right_output_path


def update_chunks(chunks, start):
    for chunk in chunks:
        chunk["timestamp"][0] += start
        chunk["timestamp"][1] += start
    return chunks


if __name__ == "__main__":
    stereo_file_path = "recording_103976.mp3"  # Path to your stereo audio file
    left_output_path = "left_channel.wav"  # Path for the left channel mono file
    right_output_path = "right_channel.wav"  # Path for the right channel mono file

    split_stereo_to_mono(stereo_file_path, "temp_data")
