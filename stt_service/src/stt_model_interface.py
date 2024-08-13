from src.stt_models.w2v_bert_interface import Wav2VecPipeline
from src.stt_models.whisper_interface import WhisperPipeline


def get_stt_model(model_type, model_paths):
    if model_type == "w2v_bert":
        return Wav2VecPipeline(model_paths["w2v_bert"], model_paths["w2v_tokenizer"])
    elif model_type == "whisper":
        return WhisperPipeline(model_paths["whisper"])
    else:
        raise ValueError("Invalid model type")


class STT:
    def __init__(self, model_type, model_paths):
        self.model_type = model_type
        self.model_path = model_paths
        self.stt_model = get_stt_model(model_type, model_paths)

    def transcribe(self, audio_path, language=None):
        return self.stt_model.transcribe(audio_path, language)

    def transcribe_stream(self, audio_stream):
        raise NotImplementedError("Subclass must implement abstract method")
