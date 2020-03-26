import multiprocessing as mp

_func = None

_lock = mp.Lock()

__all__ = ['map']


def worker_init(func):
    global _func
    _func = func


def worker(x):
    return _func(x)


def map(func, iterable):
    with _lock:
        with mp.Pool(None, initializer=worker_init, initargs=(func,)) as p:
            return p.map(worker, iterable)


if __name__ == "__main__":

    l = [1 + x for x in range(10)]

    alpha = 12.5

    def some_func(x):
        return x ** alpha

    res = map(some_func, l)

    print(res)
