from transformers import pipeline
import torch


class WhisperPipeline:
    def __init__(self, model_path):
        self.sampling_rate = 16_000
        self.model_path = model_path
        if torch.cuda.is_available():
            device = "cuda:0"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            torch_dtype=torch_dtype,
            device=device,
            return_timestamps=True,
            return_language=True,
            generate_kwargs={"task": "transcribe", "forced_decoder_ids": None},
        )
        # self.pipe.model.config.forced_decoder_ids = None

    def transcribe(self, audio_path, language=None):
        # forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        if language:
            return self.pipe(
                audio_path,
                chunk_length_s=0,
                # stride_length_s=0,
                generate_kwargs={
                    "task": "transcribe",
                    "language": language,
                    "forced_decoder_ids": None,
                },
            )
        return self.pipe(
            audio_path,
            chunk_length_s=0,
            # stride_length_s=0,
            generate_kwargs={"task": "transcribe", "forced_decoder_ids": None},
        )
