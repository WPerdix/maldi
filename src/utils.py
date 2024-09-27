from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable
from tqdm import tqdm

import multiprocessing as mp


def f_star(f_args):
    # Utility function for using mp.starmap with tqdm
    # f_args contains a tuple of (f, args)
    f = f_args[0]
    args = f_args[1:]
    return f(*args)


def f_star_i(f_args):
    # f_args contains a tuple of (f, args)
    f = f_args[0]
    args = f_args[1: -1]
    i = f_args[-1]
    return i, f(*args)


def execute_parallel_mp(pool: mp.Pool, f: Callable, *, args: list): # type: ignore
    tqdm(pool.imap_unordered(f_star, [(f, *arg) for arg in args]), total=len(args))


def execute_indexed_parallel_mp(pool: mp.Pool, f: Callable, *, args: list): # type: ignore
    return list(tqdm(pool.imap_unordered(f_star_i, [(f, *arg, i) for i, arg in enumerate(args)]), total=len(args)))


def execute_indexed_parallel(
    func: Callable, *, args: list, tqdm_args: dict = None
) -> list:
    if tqdm_args is None:
        tqdm_args = {}

    results = [None for _ in range(len(args))]
    with ThreadPoolExecutor() as executor:
        with tqdm(total=len(args), **tqdm_args) as pbar:
            futures = {executor.submit(func, *arg): i for i, arg in enumerate(args)}
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                pbar.update(1)

    return results


def execute_parallel(func: Callable, *, args: list, tqdm_args: dict = None):
    if tqdm_args is None:
        tqdm_args = {}

    with ThreadPoolExecutor() as executor:
        with tqdm(total=len(args), **tqdm_args) as pbar:
            futures = {executor.submit(func, *arg): i for i, arg in enumerate(args)}
            for _ in as_completed(futures):
                pbar.update(1)

