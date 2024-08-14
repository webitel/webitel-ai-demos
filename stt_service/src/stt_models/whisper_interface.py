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
        # self.pipe.model.config.forced_decoder_ids = self.pipe.tokenizer.get_decoder_prompt_ids(language="uk",task="transcribe")
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
                    "no_speech_threshold": 0.5,
                    "logprob_threshold": -1,
                    "temperature": (0.0, 0.2),
                    # "num_beams": 5,
                    # force model to generate new input if there is a lot of repetition with higher temperature
                    "compression_ratio_threshold": 2,
                },
            )
        return self.pipe(
            audio_path,
            chunk_length_s=0,
            # stride_length_s=0,
            generate_kwargs={
                "task": "transcribe",
                "forced_decoder_ids": None,
                "no_speech_threshold": 0.5,
                "logprob_threshold": -1,
                "temperature": (0.0, 0.2),
                # "num_beams": 5,
                "compression_ratio_threshold": 2,
            },
        )
