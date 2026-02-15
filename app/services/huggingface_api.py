import os
from typing import List, Optional
from huggingface_hub import InferenceClient
from app.core.config import Config


# client untuk call HuggingFace inference API untuk embedding.
# menggunakan InferenceClient dengan sentence_similarity.
class HuggingFaceEmbeddingAPI:
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        # API key dari parameter, env var, atau default dari Config
        self.api_key = api_key or Config.get_hf_token()
        # model name dari parameter, env var, atau default dari Config
        self.model_name = model_name or Config.get_embedding_model_name()
        
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=self.api_key,
        )
        
        if not self.api_key:
            print("[WARNING] HF_TOKEN or HUGGINGFACE_API_KEY not set. Some features may not work.")
    
    # get similarity scores antara source_sentence dan list of sentences.
    def get_similarity_scores(self, source_sentence: str, sentences: List[str]) -> List[float]:

        try:
            # sentence_similarity(source_sentence, other_sentences, model=...)
            result = self.client.sentence_similarity(
                source_sentence,
                sentences,
                model=self.model_name
            )
            return result
        except Exception as e:
            print(f"[ERROR] HuggingFace API error: {e}")
            raise