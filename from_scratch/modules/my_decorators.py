"""Betthauser, J. 2025 - This module contains various decorators that can be used to enhance the functionality of functions.
These decorators include rate limiting, debugging, type enforcement, retrying on failure, and recording execution time.

Note: by design, this program will crash at the end to demonstrate the rate_limit() decorator.
"""

import time
import random
from functools import wraps
from typing import Callable


## decorator
def rate_limit(num_allowed_calls: int = 2, period_seconds: float = 5) -> Callable:
    """decorator function that limits the number of calls to a function

    Args:
        num_allowed_calls (int, optional): _description_. Defaults to 2.
        period_seconds (float, optional): _description_. Defaults to 5.
    """

    def decorator(func: Callable) -> Callable:
        last_calls = []

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_calls
            now = time.time()
            last_calls = [call_time for call_time in last_calls if now - call_time < period_seconds]
            if len(last_calls) > num_allowed_calls:
                raise RuntimeError(f"Rate limit for function '{func.__name__}()' exceeded.")
            last_calls.append(now)
            return func(*args, **kwargs)

        return wrapper

    return decorator


## decorator
def debug(func: Callable) -> Callable:
    """decorator function that prints the function name, args, kwargs, and result"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Calling function '{func.__name__}()'")  # args:{args}, kwargs:{kwargs})'")
        result = func(*args, **kwargs)
        print(f"Result: {result}")

    return wrapper


## decorator
def type_enforce(*expected_types: tuple) -> Callable:
    """decorator function that enforces the types of the arguments passed to a function"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for arg, exp_type in zip(args, expected_types):
                if not isinstance(arg, exp_type):
                    raise TypeError(f"Expected {exp_type} but got {type(arg)}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


## decorator
def retry(retries: int = 3, exception: Exception = Exception, delay: float = 1) -> Callable:
    """decorator function that retries a function if it raises an exception"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < retries:
                try:
                    return func(*args, **kwargs)
                except exception as e:
                    attempts += 1
                    print(
                        f"function '{func.__name__}(args:{args}, kwargs:{kwargs})' failed attempt {attempts} of {retries}"
                    )
                    time.sleep(delay)
            raise exception

        return wrapper

    return decorator


## decorator
def record_time(func: Callable) -> Callable:
    """decorator function that records the time taken by a function to execute"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        tstart = time.perf_counter()
        result = func(*args, **kwargs)
        print(
            f"function '{func.__name__}()' took {time.perf_counter()-tstart:.9f} seconds."
        )  # args:{args}, kwargs:{kwargs})'
        return result

    return wrapper


###############################################################
###############################################################


@rate_limit(num_allowed_calls=2, period_seconds=5)
def _fetch_data():
    print("fetching remote data....")
    print("using lots of resources...")
    time.sleep(0.2)
    print("receiving tons of data for awhile...")


@debug
@type_enforce(int, int)
def _add(a: int, b: int):
    return a + b


@record_time
def _process_input(n):
    return [i**2 for i in range(1, n + 1)]


@retry(retries=10, exception=ValueError)
def _error_prone_function():
    if random.random() < 0.6:
        raise ValueError("Error!!!!")
    else:
        print("Success")


@record_time
def main():
    print(_process_input(25))

    _error_prone_function()

    _add(10, 20)

    try:
        print(_add("10", "20"))
    except Exception as e:
        print(e)

    for i in range(10):
        _fetch_data()
        print("~~~~~~~~~~~~~~~~~~~~~")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"Program took {time.perf_counter()-t0:.3f} seconds.")
