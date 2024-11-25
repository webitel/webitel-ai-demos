import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
import librosa

# https://github.com/SYSTRAN/faster-whisper/issues/935
from faster_whisper import WhisperModel, BatchedInferencePipeline


class FasterWhisper:
    def __init__(self, model_path, batched=True, normalize=True) -> None:
        self.batched = batched
        self.normalize = normalize
        self.model = WhisperModel(
            model_path, device="cuda", compute_type="float16", num_workers=3
        )
        if batched:
            self.batched_model = BatchedInferencePipeline(model=self.model)

    def transcribe(self, audio_path, language=None) -> tuple[list[dict[str, str]], str]:
        if self.batched:
            segments, info = self.batched_model.transcribe(
                audio_path, batch_size=16, language=language
            )
        else:
            segments, _ = self.model.transcribe(
                audio_path, beam_size=1, language=language
            )
        segments = list(segments)  # The transcription will actually run here.
        text = ""
        for segment in segments:
            text += segment.text + " "

        if self.normalize:
            text = text.replace(",", "").lower().strip()

        result = [{"text": text}]

        return result, language


class WhisperPipeline:
    def __init__(self, model_path="openai/whisper-medium", device="cpu"):
        self.sampling_rate = 16_000
        self.model_path = model_path
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.device = device

        if torch.cuda.is_available() and device == "cuda":
            device = "cuda:0"
            torch_dtype = torch.float16
        else:
            device = "cpu"
            torch_dtype = torch.float32
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype
        )
        self.model = self.model.to(device)

    def transcribe(self, audio_path, language=None):
        # forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task=task)
        y, sr = librosa.load(audio_path, sr=16_000)
        inputs = self.processor(
            y, return_tensors="pt", truncation=False, sampling_rate=16_000
        )
        if self.device == "cpu":
            pass
        elif self.device == "cuda":
            inputs = inputs.to(self.device, torch.float16)

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


# curl -X 'GET' \
#   'https://cloud.webitel.ua/api/call_center/audit/forms/2' \
#   -H 'accept: application/json' \
#   -H "x-webitel-access: ekxyxpikp7yjtfeih97ddiauyr"

# curl -X 'GET' \
#   'https://dev.webitel.com/api/call_center/audit/forms/2' \
#   -H 'accept: application/json'

# access_token=n4x5n3m9e3yrmjnjk91xoxij7h
# base_url=https://cloud.webitel.ua/api

# curl -X 'GET' \
#   'https://cloud.webitel.ua/api/storage/transcript_file/3374213/phrases?page=0&size=100' \
#   -H 'accept: application/json' \
#   -H "x-webitel-access: ekxyxpikp7yjtfeih97ddiauyr"

#   curl -X 'GET' \
#   'https://cloud.webitel.ua/api/call_center/audit/forms/6' \
#   -H 'accept: application/json'
#   -H "x-webitel-access: ekxyxpikp7yjtfeih97ddiauyr"
