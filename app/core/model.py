from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from app.core.config import Config


class EmotionModel:
    def __init__(self, model_path: str = None, max_length: int = None):
        # Max length dari parameter, env var, atau default dari Config
        self.max_length = max_length or Config.get_max_length()
        
        # Jika model_path tidak diberikan, gunakan default dari Config
        if not model_path:
            model_path = Config.get_model_path()
        
        # Cek apakah model_path adalah path lokal yang ada
        is_local_path = model_path and os.path.exists(model_path)
        
        if is_local_path:
            # Load dari path lokal
            print(f"[INFO] Loading model dari path lokal: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            # Load dari HuggingFace Hub
            # Jika model_path terlihat seperti path lokal tapi tidak ada, fallback ke model ID
            if model_path and (model_path.startswith('./') or model_path.startswith('/') or '\\' in model_path):
                print(f"[WARNING] Path lokal '{model_path}' tidak ditemukan, menggunakan model ID dari config: {Config.get_model_path()}")
                model_path = Config.get_model_path()
            
            print(f"[INFO] Loading model dari HuggingFace Hub: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Coba load dengan safetensors dulu
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    use_safetensors=True
                )
            except Exception as e:
                print(f"[WARNING] Gagal load dengan safetensors, mencoba tanpa safetensors: {e}")
                # Fallback tanpa safetensors
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    use_safetensors=False
                )
        
        self.model.eval()

    def predict(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        confidence, predicted_class = torch.max(probs, dim=1)
        label = self.model.config.id2label[predicted_class.item()]

        return {
            "label": label,
            "confidence": round(confidence.item(), 4)
        }