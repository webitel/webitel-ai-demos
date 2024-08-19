import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
import librosa


class WhisperPipeline:
    def __init__(self, model_path):
        self.sampling_rate = 16_000
        self.model_path = model_path

        self.processor = AutoProcessor.from_pretrained("openai/whisper-medium")

        if torch.cuda.is_available():
            device = "cuda:0"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-medium", torch_dtype=torch_dtype
        )
        self.model = self.model.to(device)

    def transcribe(self, audio_path, language=None):
        # forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task=task)
        y, sr = librosa.load(audio_path, sr=16_000)
        inputs = self.processor(
            y, return_tensors="pt", truncation=False, sampling_rate=16_000
        )
        inputs = inputs.to("cuda", torch.float16)

        generate_kwargs = {
            "forced_decoder_ids": None,
            "no_speech_threshold": 0.5,
            "logprob_threshold": -1,
            "temperature": (0.0, 0.2),
            # "num_beams": 5,
            "compression_ratio_threshold": 2,
        }
        # Need return_segments because of this
        # https://github.com/huggingface/transformers/issues/31942
        if language:
            output = self.model.generate(
                **inputs,
                return_timestamps=True,
                return_segments=True,
                language=language,
                task="transcribe",
                **generate_kwargs,
            )
        else:
            output = self.model.generate(
                **inputs,
                return_timestamps=True,
                return_segments=True,
                task="transcribe",
                **generate_kwargs,
            )
            # pred_lang = self.processor.batch_decode(output['sequences'][:, 1:2], skip_special_tokens=False)

        result = self.processor.batch_decode(
            output["sequences"], skip_special_tokens=True, output_offsets=True
        )

        for i in range(len(result[0]["offsets"])):
            result[0]["offsets"][i]["timestamp"] = (
                output["segments"][0][i]["start"].item(),
                output["segments"][0][i]["end"].item(),
            )
        return result, language
