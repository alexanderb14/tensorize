import itertools
import multiprocessing
import os
from enum import Enum

import tqdm
from absl import flags
from timeit import default_timer as timer

from common_helpers import *
from mlir_helpers import *


FLAGS = flags.FLAGS

all_stats = {
    "stubs_pattern_checked": 0,
    "sketches_rejected_not_simplifying": 0,
    "sketches_enumerated": 0,
    "sketches_pattern_matched": 0,
    "sketches_rejected_shape_mismatch": 0,
    "sketches_rejected_no_unknowns": 0,
    "sketches_rejected_no_intersection": 0,
    "sketches_rejected_nones_in_tensor": 0,
    "sketches_rejected_no_solution": 0,
    "sketches_solved": 0,
    "synthesis_time": 0.0,
    "sketch_construction_time": 0.0,
    "num_backtracks": 0,
    "sketches_explored": 0,
    "considered_sketches": 0,
}


def create_fresh_stats():
    return {k: 0 for k in all_stats}


def add_stats(stats):
    global all_stats
    for k in all_stats:
        all_stats[k] += stats[k]


def print_stats():
    global all_stats
    print("Stats:")
    for k in all_stats:
        print(k, all_stats[k])


class SketchType(Enum):
    MidSketch = 1
    LeafSketch = 2


class Stub:
    def __init__(self, ir, original):
        self.ir = ir
        self.original = str(original)


class Sketch:
    def __init__(self, stub, arg_idxs, expression, unknown_idxs):
        self.stub = stub
        self.arg_idxs = arg_idxs
        self.expression = expression
        self.unknown_idxs = unknown_idxs

        with Context():
            mlir_synth.register_dialects()
            mlir_synth.register_passes()

            ir = Module.parse(self.stub.ir)
            fn = get_function(ir)

            self.num_args = len(fn.arguments)

    def __repr__(self):
        return (
            "Sketch\n------------------\n"
            + "IR\n"
            + str(self.stub.ir)
            + "\n"
            + "Python\n"
            + str(self.stub.py)
            + "\n"
            + "Arg idxs\n"
            + str(self.arg_idxs)
            + "\n\n"
            + "Expression\n"
            + str(self.expression)
            + "\n"
        )


class SolutionSketch:
    def __init__(self, inputs, sketch, matched=False):
        self.inputs = inputs
        self.sketch = sketch
        self.operands = []
        self.matched = matched

        if inputs is None:
            self.inputs = []
            self.simp_score = 0
        else:
            self.simp_score = get_simp_score_nd(inputs)

    def __repr__(self):
        return (
            "SolveSketchResult\n------------------\n"
            + "simp_score\n"
            + str(self.simp_score)
            + "\n"
            + "Sketch\n"
            + str(self.sketch)
            + "\n"
        )


class Candidate:
    def __init__(self, root_sol_sk, frontier=None):
        self.root_sol_sk = root_sol_sk

        if frontier is None:
            self.frontier = [root_sol_sk]
        else:
            self.frontier = frontier

    def to_str(self):
        def indent(text, level=1):
            indent = "    " * level
            return indent + text.replace("\n", "\n" + indent)

        def helper(sol_sk, level=0):
            level += 1
            if sol_sk.operands:
                text = (
                    str(sol_sk.sketch.stub.original)
                    + "\n"
                    + str(sol_sk.inputs)
                    + "\n\n"
                    + "\n".join([helper(operand, level) for operand in sol_sk.operands])
                )

                return indent(text)
            return indent(
                str(sol_sk.sketch.stub.original) + "\n" + str(sol_sk.inputs) + "\n"
            )

        return (
            "Candidate\n------------------\n"
            + "Root solution sketch\n"
            + helper(self.root_sol_sk)
        )


def match_program_args(program, modules):
    """
    Returns for each stub a list of argument lists. Each argument list contains
    indices of matching original functions arguments
    """
    arg_idxs_all = []

    with Context(), Location.unknown():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        m = Module.parse(program)
        prog_arg_types = get_arg_types(m)
        prog_arg_types = [get_as_tensor_type(ty) for ty in prog_arg_types]

        for module in modules:
            m = Module.parse(module)
            sk_arg_types = get_arg_types(m)

            matching_prog_arg_idxs = [[] for _ in range(len(sk_arg_types))]
            for sk_arg_idx, sk_arg_type in enumerate(sk_arg_types):
                for prog_arg_idx, prog_arg_type in enumerate(prog_arg_types):
                    if prog_arg_type == sk_arg_type:
                        matching_prog_arg_idxs[sk_arg_idx] += [prog_arg_idx]
            arg_idxs_all.append(matching_prog_arg_idxs)

    return arg_idxs_all


def inject_unknowns_and_run(program, arg_symbols):
    """
    Injects unknowns in the place of each argument and runs the program
    """
    arg_vectors = []
    unknown_idxs = []

    for arg_idx, arg_symbol in enumerate(arg_symbols):
        arg_vector = []
        for arg_idx_2 in range(len(arg_symbols)):
            sym = arg_symbols[arg_idx_2]
            sym_shape = get_shape_nd(sym)

            if arg_idx == arg_idx_2:
                unknown = get_symbols_nd("X0", sym_shape)

                arg_vector += [unknown]
                unknown_idxs += [arg_idx_2]
            else:
                arg_vector += [sym]
        arg_vectors += [arg_vector]

    expressions = []
    for arg_vector in arg_vectors:
        expressions += [run(program, arg_vector)]
    return expressions, unknown_idxs


def gen_sketch(frag, arg_idxs_list, arg_symbols):
    """
    Returns sketches with injected unknowns. E.g. for a stub with arguments (A, B, C)
    will be (X, B, C), (A, X, C), (A, B, X)
    """
    mid_sketches = []

    arg_idxs_combinations = list(itertools.product(*arg_idxs_list))

    for arg_idxs in arg_idxs_combinations:
        sketch_arg_symbols = [arg_symbols[i] for i in arg_idxs]

        # Run the program with and get the expression and the unknowns
        expressions, unknown_idxs = inject_unknowns_and_run(frag.py, sketch_arg_symbols)

        for expression, unknown_idx in zip(expressions, unknown_idxs):
            # Create a sketch
            sketch = Sketch(frag, arg_idxs, expression, [unknown_idx])
            mid_sketches.append(sketch)
    return mid_sketches


def gen_sketch_full(frag):
    """
    Returns a sketch with all unknowns injected. E.g. for (A, B, C), it will be (X0_, X1_, X2_)
    """
    with Context():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        # Create unknowns in the shape of the arguments
        module = Module.parse(frag.ir)
        arg_types = get_arg_types(module)
        arg_shapes = [ShapedType(ty).shape for ty in arg_types]

        unknowns = []
        for idx, arg_shape in enumerate(arg_shapes):
            unknown = get_symbols_nd("X%d" % idx, arg_shape)
            unknowns += [unknown]

        # Run the program with the unknowns as arguments and get the expression
        try:
            expression = run(frag.py, unknowns)
        except TypeError as e:
            print(e)
            return None

        # Create a sketch
        sketch = Sketch(frag, [], expression, list(range(len(unknowns))))
        return sketch


def gen_leafsketch(stub, arg_idxs_list, arg_symbols):
    """
    Returns leaf sketches, wich are sketches without holes. E.g. for a stub with arguments (A, B, C)
    """
    leaf_sketches = []

    arg_idxs_combinations = list(itertools.product(*arg_idxs_list))

    for arg_idxs in arg_idxs_combinations:
        sketch_arg_symbols = [arg_symbols[i] for i in arg_idxs]

        # Run the program with and get the expression and the unknowns
        expression = run(stub.py, sketch_arg_symbols)

        # Create a sketch
        sketch = Sketch(stub, arg_idxs, expression, [])
        leaf_sketches.append(sketch)
    return leaf_sketches


def dedup_sketches(sketches):
    """
    Deduplicates sketches based on the expression, keeping the one with the smallest number of arguments
    """

    # Create a list of tuples (expression, sketch) in parallel
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(os.cpu_count(), initializer=init_args) as pool:
        expr_strs = smart_map(
            str,
            [s.expression for s in sketches],
            desc="Deduplicating sketches - Creating expressions",
            spawn=True,
        )

    tuples = list(zip(expr_strs, sketches))

    # Keep the sketch with the smallest number of arguments
    sketches_dedup = {}
    for expr_str, sketch in tqdm.tqdm(
        tuples, desc="Deduplicating sketches - Deduplicating"
    ):
        identifier = expr_str

        if identifier in sketches_dedup:
            if sketch.num_args < sketches_dedup[identifier].num_args:
                sketches_dedup[identifier] = sketch
        else:
            sketches_dedup[identifier] = sketch

    return list(sketches_dedup.values())


def create_sketches(
    program,
    arg_symbols,
    ops,
    constants,
    target,
    max_depth=1,
    sketch_types=[SketchType.MidSketch, SketchType.LeafSketch],
):
    stubs = target.get_stubs(program, max_depth, ops, constants)
    arg_idxs_list = match_program_args(program, [f.ir for f in stubs])

    # Create python representation
    pys = smart_map(
        hlo_to_python,
        [f.ir for f in stubs],
        desc="Creating sketches - Converting to Python",
        spawn=True,
    )

    for py, frag in tqdm.tqdm(
        zip(pys, stubs),
        total=len(stubs),
        desc="Creating sketches - Converting to Python (2)",
    ):
        frag.py = py

    # Add sketches
    mid_sketches = []
    if SketchType.MidSketch in sketch_types:
        mid_sketches = smart_starmap(
            gen_sketch,
            zip(stubs, arg_idxs_list, itertools.repeat(arg_symbols)),
            len(stubs),
            desc="Creating sketches - Adding sketches",
            spawn=True,
        )

        ret = smart_map(
            gen_sketch_full,
            stubs,
            desc="Creating sketches - Adding full sketches",
            spawn=True,
        )

        mid_sketches += [s for s in ret if s]
        mid_sketches = dedup_sketches(mid_sketches)

    # Add leaf sketches
    leaf_sketches = []
    if SketchType.LeafSketch in sketch_types:
        leaf_sketches = smart_starmap(
            gen_leafsketch,
            zip(stubs, arg_idxs_list, itertools.repeat(arg_symbols)),
            len(stubs),
            desc="Creating sketches - Adding leaf sketches",
            spawn=True,
        )

        leaf_sketches = dedup_sketches(leaf_sketches)

    return mid_sketches, leaf_sketches


def solve_sketch(task):
    stats = create_fresh_stats()
    stats["sketches_enumerated"] += 1

    source_expression, sketch = task
    if FLAGS.debug:
        print_green("sketch.original: " + str(sketch.stub.original))
        print_green("sketch.ir: " + str(sketch.stub.ir))
        print("sketch.expression: " + str(sketch.expression))

    source_expression = reparse_without_eval_nd(source_expression)

    if get_shape_nd(source_expression) != get_shape_nd(sketch.expression):
        stats["sketches_rejected_shape_mismatch"] += 1
        if FLAGS.debug:
            print(
                "Reject: Shapes do not match: ",
                get_shape_nd(source_expression),
                get_shape_nd(sketch.expression),
            )
        return [], stats
    if not has_unknowns_nd(sketch.expression):
        stats["sketches_rejected_no_unknowns"] += 1
        if FLAGS.debug:
            print("Reject: No unknowns")
        return [], stats

    match_solution = pattern_match_nd(source_expression, sketch.expression)
    if match_solution:
        stats["sketches_pattern_matched"] += 1
        # print("Matched Sketch: ", sketch.ir)
        print("Source expression:         ", source_expression)
        print("Matched Sketch expression: ", sketch.expression)
        # print("Match solution: ", match_solution)
        match_solution_list = [
            match_solution[k] for k in sorted(match_solution.keys(), reverse=True)
        ]
        return [SolutionSketch(match_solution_list, sketch, True)], stats

    if not is_any_intersection_nonempty_nd(source_expression, sketch.expression):
        stats["sketches_rejected_no_intersection"] += 1
        if FLAGS.debug:
            print("Reject: No intersection")
        return [], stats

    # Equate the source expression and the sketch expression and solve for the unknowns.
    unknowns = get_unknowns_nd(sketch.expression)
    expression = substract_symbols_nd(source_expression, sketch.expression)

    if has_nones_in_nd(expression):
        stats["sketches_rejected_nones_in_tensor"] += 1
        if FLAGS.debug:
            print("Reject: Nones in expression")
        return [], stats

    solution_shape = get_arg_shape(sketch.stub.ir, sketch.unknown_idxs[0])
    solutions = solve_nd(expression, unknowns, solution_shape)

    if FLAGS.debug:
        print("solutions: ", solutions)

    if solutions == None:
        stats["sketches_rejected_no_solution"] += 1
        return [], stats

    for solution in solutions:
        if solution == None or has_nones_in_nd(solution):
            stats["sketches_rejected_no_solution"] += 1
            return [], stats

    stats["sketches_solved"] += 1

    return [SolutionSketch([s for s in solutions], sketch)], stats


def solve_sketches(spec, sketches):
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(os.cpu_count(), initializer=init_args) as pool:
        tasks = [(spec, sketch) for sketch in sketches]
        if FLAGS.debug:
            solutions_nested = []
            stats = []
            for task in tasks:
                solutions, stat = solve_sketch(task)
                solutions_nested.append(solutions)
                stats.append(stat)
        else:
            solutions = tqdm.tqdm(
                pool.imap_unordered(solve_sketch, tasks),
                total=len(tasks),
                desc="Solving",
            )
            solutions = list(solutions)

            solutions_nested = [s[0] for s in solutions if len(s) == 2]
            stats = [s[1] for s in solutions if len(s) == 2]

        solutions = list(itertools.chain.from_iterable(solutions_nested))

        stats = {k: sum(d[k] for d in stats) for k in stats[0]}

    global all_stats
    all_stats["considered_sketches"] += len(solutions)

    add_stats(stats)

    return solutions


def contains_solution(explored, solution):
    for e in explored:
        if is_equal_nd(e, solution):
            return True
    return False


def is_sketch_equal_spec(sketch, spec, arg_symbols):
    if get_shape_nd(spec) != get_shape_nd(sketch.expression):
        return False

    sketch_arg_symbols = [arg_symbols[i] for i in sketch.arg_idxs]
    if sketch.num_args != len(sketch_arg_symbols):
        return False

    global all_stats
    all_stats["considered_sketches"] += 1
    all_stats["stubs_pattern_checked"] += 1

    sketch_expression = run(sketch.stub.py, sketch_arg_symbols)
    is_equal = is_equal_nd(spec, sketch_expression, True)
    return is_equal


def filter_and_prioritize(sol_sks, simp_score_prev, with_print=False):
    sketches_before = len(sol_sks)
    if with_print:
        print("Before: ", sketches_before)
    # Filter solutions that have more free symbols than the spec
    sol_sks = [
        x for x in sol_sks if x.simp_score < simp_score_prev or x.matched is True
    ]
    sketches_after = len(sol_sks)
    if with_print:
        print("After spec size filtering: ", sketches_after)

    global all_stats
    all_stats["sketches_rejected_not_simplifying"] += sketches_before - sketches_after

    # Filter solutions that are duplicates
    solution_str_to_solution = {}
    for solution in sol_sks:
        solution_str = str(solution.inputs)
        if solution_str in solution_str_to_solution:
            continue
        solution_str_to_solution[solution_str] = solution
    sol_sks = list(solution_str_to_solution.values())
    if with_print:
        print("After duplicate filtering: ", len(sol_sks))

    # Prioritize
    sol_sks = sorted(
        sol_sks,
        key=lambda x: (
            # x.matched == False,
            x.simp_score,
            max([get_max_ops_count_nd(solution) for solution in x.inputs]),
            str(
                (x.sketch.stub.original, x.sketch.arg_idxs)
            ),  # Make the order deterministic
        ),
    )

    return sol_sks


def get_constants(source_function, source_expression):
    # Constants from the source function
    static_constants_str = mlir_synth.predict_constants(source_function)
    static_constants = (
        [float(c) for c in static_constants_str.split(",")]
        if static_constants_str
        else []
    )

    # Constants from the source expression
    expr_constants = get_constants_nd(source_expression)
    expr_constants = [float(c) for c in expr_constants]

    # Combine the constants
    constants = list(set(list(expr_constants) + list(static_constants)))
    print("CONSTANTS: ", constants)
    return constants


def synthesize(source_function, ops, target):
    synth_start = timer()

    source_module = "module { %s }" % str(source_function)

    arg_symbols = affine_to_arg_symbols(source_module)

    # Run source program to get the source expression
    source_program = affine_to_python(source_module)
    source_expression = run(source_program, copy_arg_symbols(arg_symbols))
    print_symbols_nd(source_expression)

    constants = get_constants(source_function, source_expression)

    # Create sketches
    start = timer()
    mid_sks, leaf_sks = create_sketches(
        source_module, arg_symbols, ops, constants, target
    )
    sketch_construction_time = timer() - start
    all_stats["sketch_construction_time"] += sketch_construction_time

    # Check if any sketch already matches the source expression
    for sk in leaf_sks:
        if is_sketch_equal_spec(sk, source_expression, arg_symbols):
            print(sk.stub.ir)
            all_stats["synthesis_time"] += (
                timer() - synth_start - sketch_construction_time
            )
            return SolutionSketch([source_expression], sk)

    # Solve the source expression
    simp_score_prev = get_simp_score_nd([source_expression])

    explored = {}

    def explore(input, simp_score, level=0):
        global all_stats

        explored[str(input)] = False
        print("Level: ", level)
        print("Input expression: ", input)
        print("Complexity: ", get_num_unique_free_symbols_nd(input))

        if level > FLAGS.max_num_ops - 1:
            print("Reached max number of operations")
            return None

        # Check if one of the sketches matches the expression and return the
        # corresponding sketch in that case
        for sk in leaf_sks:
            if is_sketch_equal_spec(sk, input, arg_symbols):
                all_stats["sketches_pattern_matched"] += 1
                sol_sk = SolutionSketch([input], sk)
                return sol_sk

        # Otherwise, solve the sketches
        sks_to_explore = solve_sketches(input, mid_sks)
        sks_to_explore = filter_and_prioritize(sks_to_explore, simp_score)

        # Explore the sketches
        for sk in sks_to_explore:
            print(sk.sketch.stub.original)
            all_stats["sketches_explored"] += 1

            inputs = []
            for inp in sk.inputs:
                # Check if already explored and found a solution.
                if str(inp) in explored:
                    if explored[str(inp)]:
                        # Found a solution previously
                        inputs.append(explored[str(inp)])
                        continue
                    else:
                        # Did not find a solution previously
                        break

                # Explore
                res = explore(inp, sk.simp_score, level + 1)
                inputs.append(res)
                if res:
                    explored[str(inp)] = res
                else:
                    break
            if len(inputs) == len(sk.inputs) and all(inputs):
                sol_sk = SolutionSketch([sk.sketch.expression], sk.sketch)
                sol_sk.operands = inputs

                return sol_sk

        print("Could not find a solution: ", input)
        all_stats["num_backtracks"] += 1

        return None

    res = explore(source_expression, simp_score_prev)

    assert res

    all_stats["synthesis_time"] += timer() - synth_start - sketch_construction_time
    return res
