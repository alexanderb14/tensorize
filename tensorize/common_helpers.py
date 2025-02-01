import itertools
import multiprocessing
import os
import sys
import tqdm

from absl import flags
from sympy import *


NUM_CPUS = (
    int(os.environ.get("NUM_CPUS")) if "NUM_CPUS" in os.environ else os.cpu_count()
)

FLAGS = flags.FLAGS


def init_args():
    flags.FLAGS(sys.argv)


def run(func_py, inputs):
    # func_py = "from sympy import *\n" + func_py
    exec(func_py, globals())
    result = foo(*inputs)
    del globals()["foo"]

    return result


def print_green(s):
    print("\033[1;42m" + str(s) + "\033[0m")


def filter_ops_by_target(ops):
    if FLAGS.target == "hlo":
        return [
            op for op in ops if op.startswith("chlo.") or op.startswith("stablehlo.")
        ] + ["stablehlo.dot"]
    elif FLAGS.target == "numpy":
        return [op for op in ops if op.startswith("jnp.")]
    else:
        raise Exception


def smart_map(
    func,
    iterable,
    desc="",
    spawn=False,
    chunksize=1,
    parallel_threshold=1000,
    single_threaded=False,
):
    if len(iterable) < parallel_threshold or single_threaded:
        results = []
        for item in tqdm.tqdm(iterable, desc=desc):
            results.append(func(item))
        return results

    else:
        if spawn:
            ctx = multiprocessing.get_context("spawn")
        else:
            ctx = multiprocessing.get_context("fork")

        with ctx.Pool(NUM_CPUS) as pool:
            results = list(
                tqdm.tqdm(
                    pool.imap(func, iterable, chunksize=chunksize),
                    total=len(iterable),
                    desc=desc,
                )
            )
        return list(results)


def smart_starmap(
    func, iterable, length, desc="", spawn=False, chunksize=1, parallel_threshold=1000
):
    if length < parallel_threshold:
        results = []
        for item in tqdm.tqdm(iterable, desc=desc):
            results += func(*item)
        return results

    else:
        if spawn:
            ctx = multiprocessing.get_context("spawn")
        else:
            ctx = multiprocessing.get_context("fork")

        with ctx.Pool(NUM_CPUS) as pool:
            results = itertools.chain.from_iterable(
                pool.starmap(
                    func,
                    tqdm.tqdm(iterable, total=length, desc=desc),
                    chunksize=chunksize,
                )
            )
        return list(results)
