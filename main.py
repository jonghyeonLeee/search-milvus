# main.py
from fastapi import FastAPI
from app.services.milvus import MilvusService
from app.core.config import settings
from app.models.search import SearchRequest, SearchResults, InsertRequest, InsertResponse

app = FastAPI(title="Search API")
milvus_service = MilvusService()

@app.get("/health")
def health_check():
   return {
       "status": "ok",
       "config": {
           "milvus_host": settings.MILVUS_HOST,
           "milvus_port": settings.MILVUS_PORT,
           "collection_name": settings.COLLECTION_NAME,
           "vector_dim": settings.VECTOR_DIM
       }
   }

@app.post("/api/v1/insert", response_model=InsertResponse)
async def insert_data(request: InsertRequest):
    return await milvus_service.insert(
        text=request.text,
        metadata=request.metadata
    )

@app.post("/api/v1/search", response_model=SearchResults)
async def search(request: SearchRequest):
    return await milvus_service.search(
        query=request.query,
        limit=request.limit,
        threshold=request.threshold
    )