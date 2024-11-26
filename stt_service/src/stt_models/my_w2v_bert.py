from transformers import pipeline
from huggingface_hub import login
import torch


class W2V_BERT_WITH_LM:
    def __init__(self, device="cuda", hf_token=None):
        # Log in to Hugging Face Hub if a token is provided
        if hf_token:
            login(token=hf_token)

        # Initialize the pipeline
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        # Initialize the pipeline with appropriate precision
        self.pipe = pipeline(
            model="eingrid/wav2vec-ber-with-lm",
            device=device,
            return_timestamps="word",
            token=hf_token,
            model_kwargs={"torch_dtype": torch_dtype},
            # This ensures the model is loaded in the correct precision
        )

    def transcribe(
        self, audio_file_io, language=None, merging_seconds=1
    ) -> tuple[list[dict[str, str]], str]:
        # Run inference with chunking
        output = self.pipe(audio_file_io, chunk_length_s=30, stride_length_s=(4, 2))

        # Convert pipeline output format to match original format
        word_offsets = [
            {
                "word": chunk["text"],
                "start_time": chunk["timestamp"][0],
                "end_time": chunk["timestamp"][1],
            }
            for chunk in output["chunks"]
        ]

        # Get phrases and final transcription
        transcription = self.get_phrases(word_offsets, merging_seconds)
        transcription["text"] = output["text"]

        return [transcription], "none"

    def get_phrases(self, word_offsets, merging_seconds):
        transcription = {"offsets": []}
        current_phrase = []
        current_start_time = None
        current_end_time = None

        for word_data in word_offsets:
            if not current_phrase:
                # Start a new phrase
                current_phrase.append(word_data["word"])
                current_start_time = word_data["start_time"]
                current_end_time = word_data["end_time"]
            else:
                # Check if the current word can be merged into the ongoing phrase
                if word_data["start_time"] - current_end_time <= merging_seconds:
                    current_phrase.append(word_data["word"])
                    current_end_time = word_data["end_time"]
                else:
                    # Finalize the current phrase and add it to "offsets"
                    transcription["offsets"].append(
                        {
                            "timestamp": [current_start_time, current_end_time],
                            "text": " ".join(current_phrase),
                        }
                    )
                    # Start a new phrase
                    current_phrase = [word_data["word"]]
                    current_start_time = word_data["start_time"]
                    current_end_time = word_data["end_time"]

        # Add the last phrase if it exists
        if current_phrase:
            transcription["offsets"].append(
                {
                    "timestamp": [current_start_time, current_end_time],
                    "text": " ".join(current_phrase),
                }
            )

        return transcription
