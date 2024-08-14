import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
import torch.nn.functional as F
import torch


class SpeechBrainLanguageIdentification:
    # https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa
    def __init__(self, device="cuda"):
        self.language_id = EncoderClassifier.from_hparams(
            source="speechbrain/lang-id-voxlingua107-ecapa",
            savedir="tmp",
            run_opts={"device": device, "torch_dtype": torch.float16},
        )

    def detect_language(self, audio: str, languages: list[str]) -> str:
        signal = torchaudio.load(audio)[0]
        emb = self.language_id.encode_batch(signal)
        out_prob = self.language_id.mods.classifier(emb).squeeze(1)
        probabilities = F.softmax(out_prob[0], dim=-1)
        results = [
            {
                "label": self.preprocess_label(
                    self.language_id.hparams.label_encoder.decode_torch(
                        torch.tensor([idx])
                    )[0]
                ),
                "score": value.item(),
            }
            for idx, value in enumerate(probabilities)
        ]

        return self.select_most_probable(results, languages)

    def preprocess_label(self, label):
        return label.split(":")[0]

    def select_most_probable(self, results, languages):
        max_score = 0
        max_label = ""
        for result in results:
            if result["label"] in languages and result["score"] > max_score:
                max_score = result["score"]
                max_label = result["label"]
        return max_label
