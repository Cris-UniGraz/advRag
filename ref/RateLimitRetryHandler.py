import asyncio
import re
from typing import Dict, Optional, Callable, Any
from LLMRequestTracker import LLMRequestTracker

class RateLimitRetryHandler:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.request_tracker = LLMRequestTracker()

    def _extract_retry_time(self, error_message: str) -> int:
        match = re.search(r'retry after (\d+) second', error_message, re.IGNORECASE)
        return int(match.group(1)) if match else 1

    async def execute_with_retry(
        self,
        operation: Callable,
        request_params: Dict[str, Any]
    ) -> Any:
        # Verificar caché primero
        cached_response = self.request_tracker.get_response(request_params)
        if cached_response is not None:
            return cached_response

        retries = 0
        while retries < self.max_retries:
            try:
                response = await operation(request_params)
                # Almacenar en caché y retornar
                return self.request_tracker.store_response(request_params, response)
            except Exception as e:
                error_message = str(e)
                if "429" in error_message:
                    retries += 1
                    if retries >= self.max_retries:
                        raise Exception(f"Máximo de reintentos alcanzado ({self.max_retries})")
                    
                    retry_time = self._extract_retry_time(error_message)
                    print(f"Rate limit alcanzado. Reintento {retries}/{self.max_retries} después de {retry_time} segundos")
                    await asyncio.sleep(retry_time)
                else:
                    raise
