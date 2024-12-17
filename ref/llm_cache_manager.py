from typing import Any, Dict, Optional
from LLMRequestTracker import LLMRequestTracker
import logging
from metrics_manager import MetricsManager
from datetime import datetime
from query_optimizer import QueryOptimizer

class LLMCacheManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMCacheManager, cls).__new__(cls)
            cls._instance.cache = LLMRequestTracker()
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance.metrics = MetricsManager()
            cls._instance.priority_queue = {}
            cls._instance.query_optimizer = QueryOptimizer()
        return cls._instance
    
    def reset_cache(self):
        self.logger.info("Reseteando caché y métricas")
        self.cache.reset()
        self.metrics.reset_metrics()
        self.priority_queue.clear()
    
    def get_cached_response(self, request_params: Dict[str, Any], priority: int = 1) -> Optional[Any]:
        start_time = datetime.now()
        response = self.cache.get_response(request_params)
        
        if response is not None:
            self.metrics.metrics['cache_hits'] += 1
            self.logger.info(f"Cache hit - Prioridad: {priority}")
        else:
            self.metrics.metrics['cache_misses'] += 1
            self.logger.info(f"Cache miss - Prioridad: {priority}")
            
        duration = (datetime.now() - start_time).total_seconds()
        self.metrics.log_operation('cache_access', duration, response is not None)
        
        return response
    
    def cache_response(self, request_params: Dict[str, Any], response: Any, priority: int = 1) -> Any:
        self.logger.info(f"Almacenando respuesta en caché - Prioridad: {priority}")
        self.priority_queue[hash(str(request_params))] = priority
        return self.cache.store_response(request_params, response)
