from typing import List, Dict, Any, Optional
import hashlib
import json
from datetime import datetime, timedelta
import logging
from metrics_manager import MetricsManager
import numpy as np
from langchain.schema import Document

class QueryOptimizer:
    _instance = None

    def __init__(self):
        self.llm_cache = {}
        self.max_cache_size = 1000
        self.max_history_size = 10
        
    def _store_llm_response(self, query: str, response: str, language: str):
        """Almacena la respuesta completa del LLM"""
        query_hash = self._generate_query_hash(query)
        self.llm_cache[query_hash] = {
            'response': response,
            'timestamp': datetime.now(),
            'language': language
        }
        
        # Limpiar caché si excede el tamaño máximo
        if len(self.llm_cache) > self.max_cache_size:
            oldest_key = min(self.llm_cache.keys(), key=lambda k: self.llm_cache[k]['timestamp'])
            del self.llm_cache[oldest_key]
            
    def get_llm_response(self, query: str, language: str) -> Optional[str]:
        """Obtiene la respuesta cacheada del LLM"""
        query_hash = self._generate_query_hash(query)
        if query_hash in self.llm_cache:
            cache_entry = self.llm_cache[query_hash]
            if (datetime.now() - cache_entry['timestamp'] < timedelta(hours=24) and 
                cache_entry['language'] == language):
                return cache_entry['response']
        return None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QueryOptimizer, cls).__new__(cls)
            cls._instance.metrics = MetricsManager()
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance.query_history = {}
            cls._instance.embedding_cache = {}
            cls._instance.similarity_threshold = 0.85
        return cls._instance
        
    def _generate_query_hash(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
        
    def _validate_document(self, document: Document) -> Document:
        """
        Valida y asegura que un documento tenga todos los metadatos necesarios.
        """     
        if not hasattr(document, 'metadata') or document.metadata is None:
            document.metadata = {}
        
        # Asegurar campos mínimos requeridos
        required_fields = {
            'source': 'Unknown Source',
            'page': 'N/A',
            'doc_chunk': 0,
            'start_index': 0
        }
        
        for field, default_value in required_fields.items():
            if field not in document.metadata:
                document.metadata[field] = default_value
                
        return document

    def _store_query_result(self, query: str, result: Any, language: str):
        query_hash = self._generate_query_hash(query)
        
        # Convertir resultado a lista de Documents si es necesario
        if isinstance(result, str):
            result = [Document(page_content=result)]
        elif isinstance(result, dict):
            result = [Document(page_content=str(result))]
                
        # Validar cada documento
        validated_docs = []
        for doc in result:
            if isinstance(doc, Document):
                validated_doc = self._validate_document(doc)
                validated_docs.append(validated_doc)
            
        # Almacenar tanto el resultado como la respuesta LLM
        self.query_history[query_hash] = {
            'query': query,
            'result': validated_docs,
            'language': language,
            'timestamp': datetime.now(),
            'usage_count': 1
        }
        
        # También almacenar en el caché de LLM si es una respuesta string
        if isinstance(result, str):
            self._store_llm_response(query, result, language)


    def _get_cached_result(self, query: str, language: str) -> Optional[Any]:
        query_hash = self._generate_query_hash(query)
        if query_hash in self.query_history:
            entry = self.query_history[query_hash]
            if datetime.now() - entry['timestamp'] < timedelta(hours=24):
                entry['usage_count'] += 1
                return entry['result']
        return None
        
    def _store_embedding(self, text: str, model_name: str, embedding: np.ndarray):
        key = f"{text}:{model_name}"
        self.embedding_cache[key] = {
            'embedding': embedding,
            'timestamp': datetime.now()
        }
        
    def _get_embedding(self, text: str, model_name: str) -> Optional[np.ndarray]:
        key = f"{text}:{model_name}"
        if key in self.embedding_cache:
            cached = self.embedding_cache[key]
            if datetime.now() - cached['timestamp'] < timedelta(hours=24):
                return cached['embedding']
        return None
        
    async def optimize_query(self, 
                           query: str, 
                           language: str,
                           embedding_model: Any) -> Dict[str, Any]:
        start_time = datetime.now()
        
        # Verificar caché primero
        cached_result = self._get_cached_result(query, language)
        if cached_result:
            self.metrics.metrics['cache_hits'] += 1
            return {'result': cached_result, 'source': 'cache'}
            
        # Obtener o generar embedding
        query_embedding = self._get_embedding(query, str(embedding_model))
        if query_embedding is None:
            # Modificar esta línea para usar el método correcto según el tipo de modelo
            if hasattr(embedding_model, 'aembed_query'):
                query_embedding = await embedding_model.aembed_query(query)
            else:
                query_embedding = embedding_model.embed_query(query)
            self._store_embedding(query, str(embedding_model), query_embedding)
            
        # Procesar la consulta
        result = {
            'original_query': query,
            'language': language,
            'embedding': query_embedding,
            'timestamp': datetime.now()
        }
        
        self._store_query_result(query, result, language)
        self.metrics.metrics['cache_misses'] += 1
        
        # Registrar métricas
        processing_time = (datetime.now() - start_time).total_seconds()
        self.metrics.metrics['optimization_time'].append(processing_time)
        self.metrics.metrics['query_optimizations'] += 1
        
        return {'result': result, 'source': 'new'}
