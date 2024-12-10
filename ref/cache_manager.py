import redis
import pickle
import numpy as np
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv
from coroutine_manager import coroutine_manager

class EmbeddingCache:
    def __init__(self):
        load_dotenv()
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            password=os.getenv('REDIS_PASSWORD', None),
            decode_responses=False
        )
        self.cache_ttl = int(os.getenv('CACHE_TTL', 86400))  # 24 horas por defecto

    def _generate_key(self, text: str, model_name: str) -> str:
        return f"embedding:{model_name}:{hash(text)}"

    def get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        key = self._generate_key(text, model_name)
        cached_embedding = self.redis_client.get(key)
        if cached_embedding:
            return pickle.loads(cached_embedding)
        return None

    def store_embedding(self, text: str, model_name: str, embedding: np.ndarray):
        key = self._generate_key(text, model_name)
        self.redis_client.setex(
            key,
            self.cache_ttl,
            pickle.dumps(embedding)
        )
    @coroutine_manager.coroutine_handler()
    async def aget_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        key = self._generate_key(text, model_name)
        cached_embedding = await coroutine_manager.execute_coroutine(
            self.redis_client.get(key)
        )
        if cached_embedding:
            return pickle.loads(cached_embedding)
        return None
