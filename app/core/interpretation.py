from app.core.explanations import EMOTION_EXPLANATIONS
from app.services.huggingface_api import HuggingFaceEmbeddingAPI


# service untuk memilih interpretasi terbaik berdasarkan semantic similarity.
# menggunakan HuggingFace InferenceClient API (hybrid approach).
class InterpretationService:    
    
    def __init__(self, hf_api: HuggingFaceEmbeddingAPI):
        self.hf_api = hf_api
        # tidak perlu cache embeddings lagi, langsung pakai API
    
    def select_explanation(self, label: str, text: str):
        """
        memilih penjelasan terbaik berdasarkan semantic similarity.
        menggunakan HuggingFace sentence_similarity API.
        
        args:
            label: Label emosi yang diprediksi (contoh: "joy")
            text: Teks input yang akan dicocokkan
        
        returns:
            dict dengan keys:
                - best_explanation: string penjelasan terbaik
                - best_score: float similarity score
                - ranking: list semua kandidat dengan score
        """
        
        # ambil kandidat penjelasan
        candidates = EMOTION_EXPLANATIONS[label]
        
        # get similarity scores via API
        # source_sentence = text input
        # sentences = semua kandidat penjelasan
        similarity_scores = self.hf_api.get_similarity_scores(
            source_sentence=text,
            sentences=candidates
        )
        
        # buat ranking dengan score
        ranking_data = list(zip(candidates, similarity_scores))
        # sort by score descending
        ranking_data.sort(key=lambda x: x[1], reverse=True)
        
        # format ranking
        ranking = [
            {
                "explanation": explanation,
                "score": float(score)
            }
            for explanation, score in ranking_data
        ]
        
        # ambil yang terbaik (urutan pertama)
        best_explanation = ranking_data[0][0]
        best_score = ranking_data[0][1]
        
        return {
            "best_explanation": best_explanation,
            "best_score": float(best_score),
            "ranking": ranking
        }