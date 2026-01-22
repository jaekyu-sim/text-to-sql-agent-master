import time
from functools import wraps
from typing import Callable, Dict, Tuple, Any

def ttl_cache(ttl_seconds: int = 300):
    def decorator(fn: Callable):
        store: Dict[Tuple, Tuple[float, Any]] = {}
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.time()
            if key in store:
                ts, val = store[key]
                if now - ts < ttl_seconds:
                    return val
            val = fn(*args, **kwargs)
            store[key] = (now, val)
            return val
        return wrapper
    return decorator
