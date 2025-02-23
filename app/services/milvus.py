from typing import Dict, Any

from pymilvus import connections, Collection, utility, CollectionSchema, FieldSchema, DataType
import numpy as np

from app.services.embedding import EmbeddingService
from app.services.preprocessor import DocumentPreprocessor
from app.models.search import SearchResponse, SearchResults, UserMetadata
from app.core.config import settings

class MilvusService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self._connect()
        self.collection = self._get_collection()
        self.preprocessor = DocumentPreprocessor()

    def _connect(self):
        """Milvus 연결 설정"""
        try:
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT
            )
            print("Milvus 연결 성공")
        except Exception as e:
            print(f"Milvus 연결 실패: {str(e)}")
            raise

    def _create_collection(self) -> Collection:
        """컬렉션 생성"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=settings.VECTOR_DIM),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]

        schema = CollectionSchema(
            fields=fields,
            description="텍스트 검색을 위한 컬렉션"
        )

        collection = Collection(
            name=settings.COLLECTION_NAME,
            schema=schema,
            using='default'
        )

        # 벡터 검색을 위한 인덱스 생성
        index_params = {
            "metric_type": "IP",  # Inner Product
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        return collection

    def _get_collection(self) -> Collection:
        """컬렉션 가져오기 또는 생성"""
        try:
            collections = utility.list_collections()
            if settings.COLLECTION_NAME in collections:
                return Collection(settings.COLLECTION_NAME)
            else:
                print(f"컬렉션 '{settings.COLLECTION_NAME}'를 생성합니다.")
                return self._create_collection()
        except Exception as e:
            print(f"컬렉션 접근 실패: {str(e)}")
            raise

    async def insert(self, text: str, metadata: UserMetadata) -> Dict[str, Any]:
        try:
            # 문서 전처리
            processed_doc = await self.preprocessor.process_document(text)

            # 임베딩 생성
            embedding = await self.embedding_service.get_embedding(processed_doc["text"])

            enhanced_metadata = {
                **metadata.dict(exclude_none=True),
                **processed_doc["metadata"]
            }

            entities = [
                [processed_doc["text"]],
                [embedding],
                [enhanced_metadata]
            ]

            insert_result = self.collection.insert(entities)
            self.collection.flush()

            return {
                "id": insert_result.primary_keys[0],
                "metadata": enhanced_metadata,
                "summary": processed_doc["metadata"]["summary"]
            }

        except Exception as e:
            print(f"데이터 삽입 실패: {str(e)}")
            raise

    async def search(self, query: str, limit: int = 10, threshold: float = 0.7) -> SearchResults:
        try:
            self.collection.load()

            # 검색어 전처리 및 임베딩
            processed_query = await self.preprocessor.process_document(query)
            query_embedding = await self.embedding_service.get_embedding(processed_query["text"])

            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10},
            }

            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                output_fields=["text", "metadata"]
            )

            search_results = []
            for hits in results:
                for hit in hits:
                    if hit.distance >= threshold:
                        search_results.append(
                            SearchResponse(
                                id=hit.id,
                                text=str(hit.text),
                                score=float(hit.distance),
                                metadata=hit.metadata,
                                summary=hit.metadata.get('summary') if hit.metadata else None
                            )
                        )

            return SearchResults(
                total=len(search_results),
                results=search_results
            )

        except Exception as e:
            print(f"검색 실패: {str(e)}")
            raise