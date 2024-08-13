import torch
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2BertForCTC,
    pipeline,
)
from transformers import Wav2Vec2BertProcessor


class Wav2VecPipeline:
    def __init__(self, model_path, tokenizer_path):
        self.sampling_rate = 16_000
        self.model_path = model_path
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            tokenizer_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            word_delimiter_token="|",
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
        model = Wav2Vec2BertForCTC.from_pretrained(model_path)
        processor = Wav2Vec2BertProcessor(
            feature_extractor=feature_extractor, tokenizer=tokenizer
        )
        # can use either word or character timestamps
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            return_timestamps="word",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def transcribe(self, audio_path, language=None):
        if language:
            return self.pipe(
                audio_path,
                chunk_length_s=30,
                stride_length_s=14,
                generate_kwargs={"language": language},
            )
        return self.pipe(audio_path, chunk_length_s=30, stride_length_s=14)
