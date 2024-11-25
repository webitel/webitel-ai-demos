import torch
import soundfile as sf
from transformers import AutoModelForCTC, Wav2Vec2BertProcessor


class W2V_BERT:
    def __init__(self, device="cuda"):
        model_name = "Yehor/w2v-bert-2.0-uk-v2.1"
        # Load the model
        self.asr_model = AutoModelForCTC.from_pretrained(model_name).to(device)
        self.processor = Wav2Vec2BertProcessor.from_pretrained(model_name)
        self.sampling_rate = 16_000
        self.device = device

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

        predicted_ids = torch.argmax(logits, dim=-1)
        predictions = self.processor.batch_decode(predicted_ids)

        return predictions[0], "uk"
