from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class DocumentEmbedder:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)
        
    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)
        
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        return self.model.encode(chunks)