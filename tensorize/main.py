import json
import os
import sys

from absl import app

from mlir_synth.dialects.func import *
from mlir_synth.ir import *

import target.hlo_target as hlo_target
import target.numpy_target as numpy_target
from array_helpers import *
from common_helpers import *
from mlir_helpers import *
from synthesis import *


FLAGS = flags.FLAGS
flags.DEFINE_string("program", None, "Path to the program to synthesize")
flags.DEFINE_string("synth_out", "", "Result file containing synthesized program")
flags.DEFINE_boolean("debug", False, "Debug mode")
flags.DEFINE_boolean("distribute", False, "Distribute loops")
flags.DEFINE_bool("noguide", False, "Don't select operations automatically")
flags.DEFINE_list("ops", [], "List of operations to use")
flags.DEFINE_string("target", "numpy", "Raising target")
flags.DEFINE_integer("max_num_ops", sys.maxsize, "Maximum number of operations")


def distribution_is_legal(program_contents):
    """Whether loop distribution preserves the final value of every argument,
    compared on the symbolic traces of the whole program. Some distributions
    are not legal, e.g. in polybench/doitgen, where the reset of the `sum`
    array in the outer loop ends up in a different nest than its uses."""
    traces = []
    for passes in (["change-sizes{sizes=Primes}"],
                   ["change-sizes{sizes=Primes}", "distribute-loops"]):
        with Context():
            mlir_synth.register_dialects()
            mlir_synth.register_passes()
            module = Module.parse(program_contents)
            PassManager.parse(",".join(passes)).run(module)
            module_str = str(module)
        traces.append(str(trace_final_args(module_str)))
    if traces[0] != traces[1]:
        print("Loop distribution changes the program's semantics, not distributing")
        return False
    return True


def main(argv):
    with open(FLAGS.program, "r") as f:
        program_contents = f.read()

    program_name = os.path.basename(FLAGS.program)

    # Create target
    target = None
    if FLAGS.target == "hlo":
        target = hlo_target.HloTarget()
    elif FLAGS.target == "numpy":
        target = numpy_target.NumpyTarget()
    else:
        raise Exception

    # Run preprocessing passes
    passes = []
    passes.append("change-sizes{sizes=Primes}")
    if FLAGS.distribute:
        passes.append("distribute-loops")
    passes.append("outline-loops")

    with Context():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        module = Module.parse(program_contents)
        pm = PassManager.parse(",".join(passes))
        pm.run(module)
        multi_output = not FLAGS.distribute and has_multi_output_functions(module)

    # An outlined loop nest returns only one of the arrays it writes. If a loop
    # nest writes several, synthesize the distributed program instead, if
    # distribution preserves the program's semantics, and fall back to this
    # one if that fails.
    if multi_output and distribution_is_legal(program_contents):
        FLAGS.distribute = True
        try:
            return main(argv)
        except AssertionError:
            print("Synthesis of the distributed program failed, retrying without loop distribution")
        finally:
            FLAGS.distribute = False

    target.initialize_ast(module)

    # Syntheszie functions
    raised_fns = []
    for func_idx, func in enumerate(module.body.operations):
        if isinstance(func, FuncOp) and "irsynth.original" in func.attributes:
            print(func)

            ops = []
            if not FLAGS.noguide:
                ops_str = mlir_synth.predict_ops(func)
                ops = ops_str.split(",")
                ops = filter_ops_by_target(ops)
                ops = list(set(ops))
            if FLAGS.ops:
                ops = FLAGS.ops

            raised_fn = synthesize(func, ops, target)
            raised_fn_ast = target.construct_function_ast(
                raised_fn, func, func_idx, program_name
            )
            raised_fns.append(raised_fn_ast)

            # print(Candidate(raised).to_str())
            print_green(raised_fn_ast)

    print()
    raised_prog = target.construct_program_ast()
    print_green(raised_prog)

    # Save to disk
    if FLAGS.synth_out:
        with open(FLAGS.synth_out, "w") as f:
            f.write(raised_prog)

    print_stats()

    print()
    print("JSON:", json.dumps(all_stats))


if __name__ == "__main__":
    app.run(main)
