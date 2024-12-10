import hashlib
import json
from typing import Any, Dict

class LLMRequestTracker:
    def __init__(self):
        self.requests = {}
    
    def reset(self):
        self.requests.clear()
    
    def _generate_hash(self, request_params: Dict[str, Any]) -> str:
        # Convertir los parámetros a una cadena JSON ordenada para consistencia
        params_str = json.dumps(request_params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def get_response(self, request_params: Dict[str, Any]) -> Any:
        cache_key = self._generate_hash(request_params)
        return self.requests.get(cache_key)
    
    def store_response(self, request_params: Dict[str, Any], response: Any) -> Any:
        cache_key = self._generate_hash(request_params)
        self.requests[cache_key] = response
        return response
