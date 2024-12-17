import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Callable, Any
from metrics_manager import MetricsManager
from llm_cache_manager import LLMCacheManager
from coroutine_manager import coroutine_manager
from rate_limit_manager import RateLimitManager

class RateLimitRetryHandler:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RateLimitRetryHandler, cls).__new__(cls)
            cls._instance.max_retries = 3
            cls._instance.llm_cache = LLMCacheManager()
            cls._instance.logger = logging.getLogger(__name__)
            cls._instance.metrics = MetricsManager()
            cls._instance.rate_limit_manager = RateLimitManager()
        return cls._instance

    @coroutine_manager.coroutine_handler()
    async def execute_with_retry(
        self, 
        operation: Callable, 
        request_params: Dict[str, Any],
        priority: int = 1
    ) -> Any:
        start_time = datetime.now()
        
        try:
            cached_response = self.llm_cache.get_cached_response(request_params, priority)
            if cached_response is not None:
                return cached_response

            response = await self.rate_limit_manager.add_request(operation, request_params)
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.log_operation('api_call', duration, True)
            return self.llm_cache.cache_response(request_params, response, priority)
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.log_operation('api_call', duration, False)
            self.logger.error(f"Error en execute_with_retry: {str(e)}")
            raise
