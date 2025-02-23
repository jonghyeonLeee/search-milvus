from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    COLLECTION_NAME: str = "search_collection"
    VECTOR_DIM: int = 384  # 벡터 차원 수

    class Config:
        env_file = ".env"

settings = Settings()