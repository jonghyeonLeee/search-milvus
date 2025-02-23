from sentence_transformers import SentenceTransformer
from app.core.config import settings

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    async def get_embedding(self, text: str) -> list:
        embedding = self.model.encode(text)
        return embedding.tolist()