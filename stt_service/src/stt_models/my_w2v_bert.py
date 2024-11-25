from transformers import AutoModelForCTC, AutoProcessor
import torch
from huggingface_hub import login
import os
import soundfile as sf

login(token=os.getenv("HUGGINGFACE_TOKEN"))


class W2V_BERT_WITH_LM:
    def __init__(self, device="cuda"):
        # Load the model and processor
        self.model = AutoModelForCTC.from_pretrained(
            "eingrid/wav2vec-ber-with-lm-credit_plus"
        )
        self.processor = AutoProcessor.from_pretrained(
            "eingrid/wav2vec-ber-with-lm-credit_plus"
        )
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.sampling_rate = 16_000

    def transcribe(
        self, audio_file_io, language=None
    ) -> tuple[list[dict[str, str]], str]:
        # Transcribe the audio
        inputs = self.processor(
            sf.read(audio_file_io)[0], sampling_rate=self.sampling_rate
        ).input_features
        features = torch.tensor(inputs).to(self.device)

        with torch.inference_mode():
            logits = self.model(features).logits

        transcription = self.processor.decode(logits[0].cpu().numpy(), beam_width=10)[
            "text"
        ]

        return transcription, "uk"
