import time
from functools import wraps
from time import time

def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = time() * 1000
        try:
            return func(*args, **kwargs)
        finally:
            end_ = (time() * 1000 - start)/36000
            print(f"Total execution time: {end_ if end_ > 0 else 0} mins")
    return _time_it