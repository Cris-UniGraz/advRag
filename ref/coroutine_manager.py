import asyncio
from typing import Any, Callable, Coroutine, TypeVar, Optional
from functools import wraps
import logging

T = TypeVar('T')

class CoroutineManager:
    def __init__(self):
        self._active_coroutines = set()
        self._loop = None
        self.logger = logging.getLogger(__name__)
        
    @property
    def loop(self):
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        return self._loop
    
    async def execute_coroutine(self, coroutine: Coroutine) -> Any:
        """
        Ejecuta una corrutina de manera segura y la registra para su seguimiento
        """
        try:
            self._active_coroutines.add(coroutine)
            result = await coroutine
            return result
        finally:
            self._active_coroutines.discard(coroutine)
            
    async def execute_with_timeout(self, coroutine: Coroutine, timeout: float) -> Any:
        """
        Ejecuta una corrutina con un tiempo límite
        """
        try:
            return await asyncio.wait_for(self.execute_coroutine(coroutine), timeout)
        except asyncio.TimeoutError:
            self.logger.warning(f"Coroutine timeout after {timeout} seconds")
            raise
            
    def ensure_future(self, coroutine: Coroutine) -> asyncio.Task:
        """
        Programa una corrutina para ejecución futura y la registra
        """
        task = self.loop.create_task(self.execute_coroutine(coroutine))
        return task
    
    async def gather_coroutines(self, *coroutines: Coroutine) -> list:
        """
        Ejecuta múltiples corrutinas en paralelo y espera todos los resultados
        """
        tasks = [self.ensure_future(coro) for coro in coroutines]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]
    
    def coroutine_handler(self, timeout: Optional[float] = None):
        """
        Decorador para manejar funciones asíncronas
        """
        def decorator(func: Callable[..., Coroutine]):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                coroutine = func(*args, **kwargs)
                if timeout is not None:
                    return await self.execute_with_timeout(coroutine, timeout)
                return await self.execute_coroutine(coroutine)
            return wrapper
        return decorator
    
    async def cleanup(self):
        """
        Limpia y cancela todas las corrutinas activas
        """
        for coroutine in self._active_coroutines:
            if isinstance(coroutine, asyncio.Task) and not coroutine.done():
                coroutine.cancel()
        self._active_coroutines.clear()

# Instancia global del gestor de corrutinas
coroutine_manager = CoroutineManager()
