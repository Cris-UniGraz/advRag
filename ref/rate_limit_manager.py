import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any
import logging
from metrics_manager import MetricsManager

class RateLimitManager:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.request_queue = asyncio.Queue()
        self.last_request_times = []
        self.metrics = MetricsManager()
        self.logger = logging.getLogger(__name__)
        
    async def add_request(self, operation: callable, request_params: Dict[str, Any]) -> Any:
        await self.request_queue.put((operation, request_params))
        return await self.process_request()
        
    async def process_request(self) -> Any:
        current_time = datetime.now()
        self.last_request_times = [t for t in self.last_request_times 
                                 if current_time - t < timedelta(minutes=1)]
        
        if len(self.last_request_times) >= self.requests_per_minute:
            wait_time = 60 - (current_time - min(self.last_request_times)).total_seconds()
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time} seconds")
                await asyncio.sleep(wait_time)
        
        operation, params = await self.request_queue.get()
        try:
            self.last_request_times.append(datetime.now())
            result = await operation(params)
            self.metrics.metrics['successful_requests'] += 1
            return result
        except Exception as e:
            self.metrics.metrics['failed_requests'] += 1
            self.logger.error(f"Error processing request: {str(e)}")
            raise
