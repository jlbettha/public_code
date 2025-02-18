import time
import random


## decorator
def rate_limit(calls, period):
    def decorator(func):
        last_calls = []

        def wrapper(*args, **kwargs):
            nonlocal last_calls
            now = time.time()
            last_calls = [
                call_time for call_time in last_calls if now - call_time < period
            ]
            if len(last_calls) > calls:
                raise RuntimeError(
                    f"Rate limit for function '{func.__name__}()' exceeded."
                )
            last_calls.append(now)
            return func(*args, **kwargs)

        return wrapper

    return decorator


## decorator
def debug(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function '{func.__name__}(args:{args}, kwargs:{kwargs})'")
        result = func(*args, **kwargs)
        print(f"Result: {result}")

    return wrapper


## decorator
def type_enforce(*expected_types):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for arg, exp_type in zip(args, expected_types):
                if not isinstance(arg, exp_type):
                    raise TypeError(f"Expected {exp_type} but got {type(arg)}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


## decorator
def retry(retries=3, exception=Exception, delay=1):
    def decorator(func):
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
def record_time(func):
    def wrapper(*args, **kwargs):
        tstart = time.perf_counter()
        result = func(*args, **kwargs)
        print(
            f"function '{func.__name__}(args:{args}, kwargs:{kwargs})' took {time.perf_counter()-tstart:.3f} seconds."
        )
        return result

    return wrapper


###############################################################
###############################################################


@rate_limit(calls=2, period=5)
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
    t0 = time.time()
    main()
    print(f"Program took {time.time()-t0:.3f} seconds.")
