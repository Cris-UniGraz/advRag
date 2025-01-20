from datetime import datetime
import logging
from typing import Dict, Any
from collections import defaultdict

class MetricsManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MetricsManager, cls).__new__(cls)
            cls._instance.reset_metrics()
            cls._instance.logger = logging.getLogger(__name__)
        return cls._instance
    
    def reset_metrics(self):
        self.metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'rate_limits': 0,
            'api_calls': 0,
            'response_times': [],
            'operation_counts': defaultdict(int),
            'errors': defaultdict(int),
            'processing_times': [],
            'document_counts': [],
            'query_lengths': [],
            'query_optimizations': 0,
            'optimization_time': [],
            'cache_hit_rate': 0,
            'query_similarity_scores': [],
            'cache_cleanups': 0,  # Añadir esta línea
            'entries_removed': 0  # También es buena idea añadir esta métrica
        }
        self.start_time = datetime.now()

   
    def log_operation(self, operation_type: str, duration: float, success: bool):
        self.metrics['operation_counts'][operation_type] += 1
        self.metrics['response_times'].append(duration)
        if not success:
            self.metrics['errors'][operation_type] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        total_time = (datetime.now() - self.start_time).total_seconds()
        avg_response_time = sum(self.metrics['response_times']) / len(self.metrics['response_times']) if self.metrics['response_times'] else 0
        
        return {
            'total_operations': sum(self.metrics['operation_counts'].values()),
            'cache_hit_rate': self.metrics['cache_hits'] / (self.metrics['cache_hits'] + self.metrics['cache_misses']) if (self.metrics['cache_hits'] + self.metrics['cache_misses']) > 0 else 0,
            'average_response_time': avg_response_time,
            'error_rate': sum(self.metrics['errors'].values()) / sum(self.metrics['operation_counts'].values()) if sum(self.metrics['operation_counts'].values()) > 0 else 0,
            'total_time': total_time
        }
    
    def log_query_optimization(self, processing_time: float, was_cached: bool):
        self.metrics['optimization_time'].append(processing_time)
        if was_cached:
            self.metrics['cache_hits'] += 1
        else:
            self.metrics['cache_misses'] += 1

