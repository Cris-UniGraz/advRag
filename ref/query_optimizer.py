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
        
    def _store_llm_response(self, query: str, response: str, language: str, sources: List[Dict] = None):
        query_hash = self._generate_query_hash(query)
        
        validated_sources = []
        
        if sources and isinstance(sources, list):
            for source in sources:
                if isinstance(source, dict):
                    validated_source = {
                        'source': source.get('source', 'Unknown Source'),
                        'page': source.get('page', 'N/A'),
                        'sheet_name': source.get('sheet_name'),
                        'page_number': source.get('page_number')
                    }
                    validated_sources.append(validated_source)

        self.llm_cache[query_hash] = {
            'response': response,
            'timestamp': datetime.now(),
            'language': language,
            'sources': validated_sources,
            'original_query': query
        }

        # Limpiar caché si excede el tamaño máximo
        if len(self.llm_cache) > self.max_cache_size:
            oldest_key = min(self.llm_cache.keys(), key=lambda k: self.llm_cache[k]['timestamp'])
            del self.llm_cache[oldest_key]

    def get_llm_response(self, query: str, language: str) -> Optional[Dict]:
        query_hash = self._generate_query_hash(query)
       
        if query_hash in self.llm_cache:
            cache_entry = self.llm_cache[query_hash]

            if (datetime.now() - cache_entry['timestamp'] < timedelta(hours=24) and 
                cache_entry['language'] == language):
                return cache_entry  # Devolver directamente la entrada del caché
        return None


    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QueryOptimizer, cls).__new__(cls)
            # Inicializar todos los atributos aquí
            cls._instance.llm_cache = {}
            cls._instance.max_cache_size = 1000
            cls._instance.max_history_size = 10
            cls._instance.metrics = MetricsManager()
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance.query_history = {}
            cls._instance.embedding_cache = {}
            cls._instance.similarity_threshold = 0.85
        return cls._instance

    def __init__(self):
        # Remover la inicialización de atributos de aquí
        pass

        
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
        
        # Extraer la respuesta y fuentes del resultado
        response = ''
        sources = []
        
        if isinstance(result, dict):
            response = result.get('response', '')
            sources = result.get('sources', [])
        
        cached_data = {
            'response': response,
            'sources': sources,
            'timestamp': datetime.now(),
            'language': language,
            'original_query': query,
            'usage_count': 0
        }
        
        self.query_history[query_hash] = cached_data
        
        # También almacenar en el caché LLM
        if response:
            self._store_llm_response(query, response, language, sources)


    def _get_cached_result(self, query: str, language: str) -> Optional[Any]:
        query_hash = self._generate_query_hash(query)
        if query_hash in self.query_history:
            entry = self.query_history[query_hash]
            if datetime.now() - entry['timestamp'] < timedelta(hours=24):
                entry['usage_count'] += 1
                return {
                    'response': entry.get('response', ''),
                    'sources': entry.get('sources', []),
                    'language': entry.get('language', language),
                    'original_query': entry.get('original_query', query),
                    'from_cache': True
                }
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
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Limpia entradas antiguas del caché basado en timestamp"""
        current_time = datetime.now()
        keys_to_remove = []
        
        # Identificar entradas antiguas
        for query_hash, entry in self.llm_cache.items():
            age = current_time - entry['timestamp']
            if age > timedelta(hours=max_age_hours):
                keys_to_remove.append(query_hash)
        
        # Eliminar entradas antiguas
        for key in keys_to_remove:
            del self.llm_cache[key]
            
        # Registrar métricas de limpieza
        self.metrics.metrics['cache_cleanups'] += 1
        self.metrics.metrics['entries_removed'] = len(keys_to_remove)

        print(f"-> Cache cleanup completed. Removed {len(keys_to_remove)} entries.")
        
        return len(keys_to_remove)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del estado actual del caché"""
        current_time = datetime.now()
        stats = {
            'total_entries': len(self.llm_cache),
            'oldest_entry': None,
            'newest_entry': None,
            'avg_age_hours': 0
        }
        
        if self.llm_cache:
            ages = [(current_time - entry['timestamp']).total_seconds() / 3600 
                    for entry in self.llm_cache.values()]
            stats.update({
                'oldest_entry': max(ages),
                'newest_entry': min(ages),
                'avg_age_hours': sum(ages) / len(ages)
            })
        
        return stats

    def cleanup_old_entries(self):
        current_time = datetime.now()
        keys_to_remove = [
            key for key, entry in self.llm_cache.items()
            if (current_time - entry['timestamp']) > timedelta(hours=24)
        ]
        for key in keys_to_remove:
            del self.llm_cache[key]
        return len(keys_to_remove)
