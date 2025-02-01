import ast
import copy
import itertools
import math

import jax
import jax.numpy as jnp
import tqdm
from jax import config

import mlir_synth.synth as mlir_synth
from mlir_synth.ir import *
from mlir_synth.passmanager import *

from synthesis import *
from target.target import Target
from mlir_helpers import *


config.update("jax_enable_x64", True)

NUMPY_GRAMMAR = [
    (jnp.sum, float, [float]),
    (jnp.max, float, [float]),
    (jnp.exp, float, [float]),
    (jnp.transpose, float, [float]),
    (jnp.sqrt, float, [float]),
    (jnp.divide, float, [float, float]),
    (jnp.add, float, [float, float]),
    (jnp.subtract, float, [float, float]),
    (jnp.multiply, float, [float, float]),
    (jnp.remainder, float, [float, float]),
    (jnp.dot, float, [float, float]),
    (jnp.tensordot, float, [float, float]),
    (jnp.equal, bool, [float, float]),
    (jnp.less, bool, [float, float]),
    (jnp.less_equal, bool, [float, float]),
    (jnp.greater, bool, [float, float]),
    (jnp.greater_equal, bool, [float, float]),
    (jnp.power, float, [float, float]),
    (jnp.where, float, [bool, float, float]),
]


class Candidate:
    def __init__(self, fn):
        self.fn = fn
        self.args = []
        self.status = "ok"

    def merge(self, others):
        """
        This function is a Python AST port of the merge function in
        Candidate.cc, which originally works on MLIR.
        """
        # Get ast of self.fn
        fn_ast = ast.parse(self.fn)
        fn_body = fn_ast.body[0].body

        mapping = {}
        var_id = 0

        # Merge other candidates arguments
        for other_idx, other in enumerate(others):
            other_ast = ast.parse(other.fn)
            other_args = other_ast.body[0].args.args
            if other_args:
                for other_arg in other_args:
                    # Create a new argument
                    arg_name = "arg%d" % len(fn_ast.body[0].args.args)
                    mapping["%d.%s" % (other_idx, other_arg.arg)] = arg_name

                    # Add the argument to the function
                    other_arg = copy.deepcopy(other_arg)
                    other_arg.arg = arg_name
                    fn_ast.body[0].args.args.append(other_arg)

                # Add the others args to this candidates
                self.args += other.args

        # Clone other candidates operations, and update the operands with the
        # mapping. Also, add the results of the other operations to the resultValues.
        resultValues = [None for _ in range(len(others))]
        for other_idx, other in enumerate(others):
            other_ast = ast.parse(other.fn)
            other_body = other_ast.body[0].body

            # assert len(other_body) == 1  # We only support one return statement
            assert isinstance(other_body[-1], ast.Return)

            for other_stmt in other_body:
                other_stmt = other_stmt.value
                if isinstance(other_stmt, ast.Name):
                    # No need to clone anything, as the candidate is just the argument

                    # Add the result to the resultValues
                    resultValues[other_idx] = mapping[
                        "%d.%s" % (other_idx, other_stmt.id)
                    ]
                else:
                    # Clone the operation
                    other_stmt = copy.deepcopy(other_stmt)
                    ## Update the operands
                    # other_stmt.args = [ast.parse(mapping["%d.%s" % (other_idx, arg.id)]).body[0].value
                    #                         for arg in other_stmt.args if isinstance(arg, ast.Name)]

                    # Create an assignment statement for the result of the operation
                    other_stmt = ast.Assign(
                        [ast.Name("v%d" % var_id, ast.Store())], other_stmt
                    )
                    var_id += 1

                    # Add the operation to the function
                    fn_ast.body[0].body.insert(var_id - 1, other_stmt)

                    # Add the result to the resultValues
                    resultValues[other_idx] = other_stmt.targets[0].id

        # Replace the operands of the return statement with the resultValues
        assert isinstance(fn_body[-1], ast.Return)
        fn_body[-1].value.args = [ast.arg(arg, None) for arg in resultValues]

        ## Remove unused arguments
        # used_args = set(
        #    [arg.id for arg in ast.walk(other_ast) if isinstance(arg, ast.Name)]
        # )
        # fn_ast.body[0].args.args = [
        #    arg for arg in fn_ast.body[0].args.args if arg.arg in used_args
        # ]

        # Transform the new function into python code
        self.fn = str(ast.unparse(ast.fix_missing_locations(fn_ast)))


def get_initial_candidates(arg_shapes, constants):
    candidates = {float: [], bool: []}

    # - Each arg
    #   - Each arg as a float
    for arg_shape in arg_shapes:
        cand = Candidate("def foo(arg0): return arg0")
        arg = jnp.full(arg_shape, 1.23, dtype=jnp.float64) if len(arg_shape) else 1.23
        cand.args = [arg]
        candidates[float].append(cand)

    #   - Each arg as a boolean
    for arg_shape in arg_shapes:
        cand = Candidate("def foo(arg0): return arg0")
        arg = jnp.full(arg_shape, True, dtype=jnp.bool_) if len(arg_shape) else True
        cand.args = [arg]
        candidates[bool].append(cand)

    # - Each arg shape
    for arg_shape in arg_shapes:
        if not len(arg_shape):
            continue

        arg = "jnp.full({}, 1.23, dtype=jnp.float64)".format(arg_shape)
        candidates[float].append(Candidate("def foo(): return {}".format(arg)))

    # - Constants
    for arg_shape in arg_shapes:
        for constant in set([0, 1] + constants):
            arg = (
                "jnp.full({}, {})".format(arg_shape, constant)
                if len(arg_shape)
                else str(constant)
            )
            candidates[float].append(Candidate("def foo(): return {}".format(arg)))

    for arg_shape in arg_shapes:
        for constant in ["True", "False"]:
            arg = (
                "jnp.full({}, {})".format(arg_shape, constant)
                if len(arg_shape)
                else str(constant)
            )
            candidates[bool].append(Candidate("def foo(): return {}".format(arg)))

    # Triangular masks
    for arg_shape in arg_shapes:
        # arg = "jnp.tril({})".format(arg_shape)
        # candidates[bool].append(Candidate("def foo(): return {}".format(arg)))

        # arg = "jnp.triu({})".format(arg_shape)
        # candidates[bool].append(Candidate("def foo(): return {}".format(arg)))

        arg = (
            "jnp.tri({}, dtype=bool)".format(arg_shape[0]) if len(arg_shape) else "True"
        )
        candidates[bool].append(Candidate("def foo(): return {}".format(arg)))

        arg = (
            "jnp.eye({}, dtype=bool)".format(arg_shape[0]) if len(arg_shape) else "True"
        )
        candidates[bool].append(Candidate("def foo(): return {}".format(arg)))

    return candidates


def compile_candidate(candidate):
    try:
        compiled = compile("global foo\n" + str(candidate.fn), "", "exec")
        exec(compiled)

        jax_fn = jax.jit(foo)
        hlo = jax_fn.lower(*candidate.args).compiler_ir(dialect="mhlo")

        if len(list(hlo.body.operations)) > 1:
            hlo = inline_functions(hlo)

        hlo_str = str(hlo)

        # Clean up
        del globals()["foo"]
        del hlo
        del jax_fn
        jax.clear_caches()

        return hlo_str
    except (ValueError, TypeError) as e:
        return None


def compile_candidates(candidates, desc="Compiling candidates"):
    ir_strs = smart_map(compile_candidate, candidates, desc=desc, spawn=True)

    ret = []
    for candidate, ir in zip(candidates, ir_strs):
        if ir is None:
            continue
        candidate.ir = ir
        ret.append(candidate)
    return ret


def run_candidate(candidate):
    arg_symbols = affine_to_arg_symbols(candidate.ir)
    py = hlo_to_python(candidate.ir)

    if py == "":
        return None

    try:
        expr = run(py, copy_arg_symbols(arg_symbols))
    except ZeroDivisionError:
        # print("ZeroDivisionError encountered")
        return None
    if expr is None:
        return None

    return expr


def run_candidates(candidates):
    expr_strs = smart_map(
        run_candidate, candidates, desc="Running candidates", spawn=True
    )

    ret = []
    for candidate, expr_str in zip(candidates, expr_strs):
        if expr_str is None:
            continue

        candidate.expr = expr_str

        ret.append(candidate)
    return ret


def simplify_candidate(candidate):
    expr = candidate.expr

    expr, status = apply_nd(expr, sympy.factor)
    if expr is None:
        return None, status

    expr, status = apply_nd(expr, sympy.simplify)
    if expr is None:
        return None, status

    return str(expr), status


def simplify_candidates(candidates):
    results = smart_map(
        simplify_candidate, candidates, desc="Simplifying candidates", spawn=True
    )

    ret = []
    for candidate, (expr, status) in zip(candidates, results):
        if expr is None:
            continue

        candidate.expr = expr
        candidate.status = status

        ret.append(candidate)
    return ret


def dedup_candidates(candidates):
    exprs = [candidate.expr for candidate in candidates]
    expr_strs = smart_map(
        str, exprs, desc="Deduplicating: Stringifying expressions", spawn=True
    )

    ret = {}
    for candidate, expr_str in tqdm.tqdm(
        zip(candidates, expr_strs), desc="Deduplicating: Deduping candidates"
    ):
        if expr_str not in ret:
            ret[expr_str] = candidate
    return list(ret.values())


def enumerate_candidates_worker(task):
    op_name, n_args, ret_type, comb = task
    new_candidates = {float: [], bool: []}

    # Attributes
    if op_name == "tensordot":
        axes = list(itertools.product([0, 1, 2], repeat=2))
        for axe in axes:
            cand = Candidate(
                "def foo(): return jnp.{}({}, axes={})".format(
                    op_name,
                    ", ".join(["unk"] * n_args),
                    str(axe),
                )
            )
            cand.merge(comb)
            new_candidates[ret_type].append(cand)

    elif op_name == "sum" or op_name == "max":
        axes = [0, 1, 2]
        for axe in axes:
            cand = Candidate(
                "def foo(): return jnp.{}({}, axis={})".format(
                    op_name,
                    ", ".join(["unk"] * n_args),
                    str(axe),
                )
            )
            cand.merge(comb)
            new_candidates[ret_type].append(cand)

    else:
        cand = Candidate(
            "def foo(): return jnp.{}({})".format(op_name, ", ".join(["unk"] * n_args))
        )
        cand.merge(comb)
        new_candidates[ret_type].append(cand)

    return new_candidates


def enumerate_candidates(arg_shapes, ops, constants, max_depth=1):
    last_candidates = get_initial_candidates(arg_shapes, constants)

    # Create candidates for operations by generating operands with
    # cartesian product of initial candidates
    new_candidates = {float: [], bool: []}

    for depth in range(max_depth):
        for op, ret_type, arg_types in ops:
            if ret_type not in new_candidates:
                new_candidates[ret_type] = []

            n_args = len(arg_types)
            combs = itertools.product(*[last_candidates[ty] for ty in arg_types])
            combs = list(combs)

            chunksize = math.ceil(len(combs) / NUM_CPUS)
            results = smart_map(
                enumerate_candidates_worker,
                [(op.__name__, n_args, ret_type, comb) for comb in combs],
                desc="Enumerating candidates for %s" % op.__name__,
                spawn=True,
                chunksize=chunksize,
            )

            for result in results:
                for dtype in result.keys():
                    new_candidates[dtype] += result[dtype]

        candidates = {}
        for dtype in last_candidates.keys():
            candidates[dtype] = compile_candidates(
                last_candidates[dtype] + new_candidates[dtype]
            )
            # candidates[dtype] = run_and_dedup_candidates(candidates[dtype])

            candidates[dtype] = run_candidates(candidates[dtype])
            # candidates[dtype] = simplify_candidates(candidates[dtype])
            candidates[dtype] = dedup_candidates(candidates[dtype])

        last_candidates = candidates

        for dtype in last_candidates.keys():
            print(
                "Depth %d: %d candidates for %s"
                % (depth, len(last_candidates[dtype]), dtype)
            )

    return list(itertools.chain.from_iterable(last_candidates.values()))


def extend_candidate_w_rhs_add(candidate):
    candidate = copy.deepcopy(candidate)

    # Skip functions that have only a single operation, which is a return
    with Context():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        ir = Module.parse(candidate.ir)

        func = get_function(ir)
        block = func.regions[0].blocks[0]

        if len(block.operations) == 1:
            return None

    # Get return shape
    with Context():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        ir = Module.parse(candidate.ir)
        ret_shape = get_return_shape(ir)

    # Create a function with a signature that includes a new argument with the
    # same type as the return type
    fn_ast = ast.parse(candidate.fn)
    fn_args = fn_ast.body[0].args.args
    fn_args.append(ast.arg("arg%d" % len(fn_args), None))

    # Remove the return, but store what it referenced in a variable
    fn_body = fn_ast.body[0].body
    assert isinstance(fn_body[-1], ast.Return)
    ret_stmt_old = ast.Assign(
        [ast.Name("v%d" % len(fn_body), ast.Store())], fn_body[-1].value
    )
    fn_body[-1] = ret_stmt_old

    # Create a jnp.add stmt that adds op_wip and the last argument
    # of the function
    add_stmt = ast.Assign(
        [ast.Name("v%d" % (len(fn_body) + 1), ast.Store())],
        ast.Call(
            func=ast.Attribute(
                value=ast.Name("jnp", ast.Load()), attr="add", ctx=ast.Load()
            ),
            args=[
                ret_stmt_old.targets[0],
                ast.Name("arg%d" % (len(fn_args) - 1), ast.Load()),
            ],
            keywords=[],
        ),
    )
    fn_body.append(add_stmt)

    # Create a return stmt
    return_stmt = ast.Return([add_stmt.targets[0]])
    fn_body.append(return_stmt)

    candidate.fn = ast.unparse(ast.fix_missing_locations(fn_ast))

    # Add arg to candidate
    arg = jnp.full(ret_shape, 1.23, dtype=jnp.float64) if len(ret_shape) else 1.23
    candidate.args += [arg]

    return candidate


def extend_candidates_w_rhs_add(candidates):
    candidates_w_rhs_add = smart_map(
        extend_candidate_w_rhs_add,
        candidates,
        desc="Extending candidates with rhs add",
        single_threaded=True,
    )

    return [c for c in candidates_w_rhs_add if c is not None]


def get_last_function(py_str):
    tree = ast.parse(py_str)
    last_func = tree.body[-1]
    return ast.unparse(last_func)


def replace_return_value(py_str):
    tree = ast.parse(py_str)
    last_func = tree.body[-1]

    # Get the value of the last function call
    assert isinstance(last_func.body[-1], ast.Return)

    last_call = None
    for stmt in last_func.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            last_call = stmt.value
    assert last_call

    # Remove the return statement
    last_func.body = last_func.body[:-2]

    # Add a return statement returning the value of the last function call
    last_func.body.append(ast.Return(value=last_call))

    return ast.unparse(last_func)


def get_fn(fn_str, fn_name):
    fn_ast = ast.parse(fn_str)
    fn_body = fn_ast.body
    for stmt in fn_body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == fn_name:
            return stmt
    return None


def get_stmts_without_return(fn):
    return [stmt for stmt in fn.body if not isinstance(stmt, ast.Return)]


def inline_ast_functions(prog_str):
    prog_ast = ast.parse(prog_str)
    fn = prog_ast.body[-1]

    for stmt in fn.body:
        if isinstance(stmt.value, ast.Call):
            # Get the called function
            called_fn_name = stmt.value.func.id
            called_fn = get_fn(prog_str, called_fn_name)

            # Map the arguments of the called function to the arguments of the caller
            mapping = {}
            for arg_idx, arg in enumerate(called_fn.args.args):
                mapping[arg.arg] = stmt.value.args[arg_idx].id

            # Build new body
            new_body = []
            for stmt_idx, called_fn_stmt in enumerate(called_fn.body):
                if isinstance(called_fn_stmt, ast.Assign) and type(
                    called_fn_stmt.value
                ) in [ast.Call, ast.Name, ast.Constant]:
                    # Rename the arguments of the called function with the names of arguments of the caller
                    # LHS
                    if stmt_idx < len(get_stmts_without_return(called_fn)) - 1:
                        # Here we're at statements before the last statement of the function
                        var_id = called_fn_name + "_var" + str(stmt_idx)
                        old_var_id = called_fn_stmt.targets[0].id
                        called_fn_stmt.targets[0].id = var_id
                        mapping[old_var_id] = var_id
                    else:
                        # Here we're at the last statement of the function
                        if isinstance(stmt, ast.Return):
                            called_fn_stmt = ast.Return(called_fn_stmt.value)
                        elif isinstance(stmt, ast.Assign):
                            called_fn_return = stmt.targets[0].id
                            called_fn_stmt.targets[0].id = called_fn_return
                        else:
                            print("ERROR: Unsupported statement type %s" % type(stmt))
                            assert False

                    # RHS
                    rhs = called_fn_stmt.value
                    if isinstance(rhs, ast.Call):
                        for arg in rhs.args:
                            if isinstance(arg, ast.Name):
                                if arg.id not in mapping:
                                    print(
                                        "ERROR: arg.id: %s not in mapping %s "
                                        % (arg.id, mapping)
                                    )
                                    assert False

                                arg.id = mapping[arg.id]
                    elif isinstance(rhs, ast.Name):
                        assert rhs.id in mapping
                        rhs.id = mapping[rhs.id]
                    elif isinstance(rhs, ast.Constant):
                        pass
                    else:
                        print("ERROR: Unsupported statement type %s" % type(rhs))
                        assert False

                    new_body.append(called_fn_stmt)

            # Add the statements of the called function to the caller
            stmt_idx = fn.body.index(stmt)
            fn.body = fn.body[:stmt_idx] + new_body + fn.body[stmt_idx + 1 :]

    return ast.unparse(fn)


def inline_stmts(prog_str):
    prog_ast = ast.parse(prog_str)
    fn = prog_ast.body[-1]

    # if len(fn.body) > 5:
    #    return prog_str

    defs = {arg.arg: arg for arg in fn.args.args}
    to_remove = []

    for stmt in fn.body:
        if isinstance(stmt, ast.Assign):
            assign_stmt = stmt

            call_stmt = assign_stmt.value
            if isinstance(call_stmt, ast.Call):
                for arg_idx, arg in enumerate(call_stmt.args):
                    if isinstance(arg, ast.Name):
                        call_stmt.args[arg_idx] = defs[arg.id]

                for target in assign_stmt.targets:
                    defs[target.id] = call_stmt
            elif isinstance(call_stmt, ast.Name):
                defs[assign_stmt.targets[0].id] = defs[call_stmt.id]
            elif isinstance(call_stmt, ast.Constant):
                defs[assign_stmt.targets[0].id] = call_stmt
            elif isinstance(call_stmt, ast.BinOp):
                if isinstance(call_stmt.left, ast.Name):
                    call_stmt.left = defs[call_stmt.left.id]
                if isinstance(call_stmt.right, ast.Name):
                    call_stmt.right = defs[call_stmt.right.id]
                defs[assign_stmt.targets[0].id] = call_stmt
            else:
                print("ERROR: Unsupported statement type %s" % type(call_stmt))
                assert False
            to_remove.append(stmt)

        elif isinstance(stmt, ast.Return):
            return_stmt = stmt

            call_stmt = return_stmt.value
            if isinstance(call_stmt, ast.Call):
                for arg_idx, arg in enumerate(call_stmt.args):
                    if isinstance(arg, ast.Name):
                        call_stmt.args[arg_idx] = defs[arg.id]
                return_stmt.value = call_stmt
            elif isinstance(call_stmt, ast.Name):
                return_stmt.value = defs[call_stmt.id]
            else:
                raise NotImplementedError

    for stmt in to_remove:
        fn.body.remove(stmt)

    return ast.unparse(fn)


def restore_sizes(prog_str, sizes):
    prog_ast = ast.parse(prog_str)
    fn = prog_ast.body[-1]

    for stmt in fn.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            for arg in stmt.value.args:
                if isinstance(arg, ast.List):
                    for ele in arg.elts:
                        if isinstance(ele, ast.Constant):
                            original_value = sizes[ele.value]
                            ele.value = original_value

    return ast.unparse(fn)


def normalize_identifiers(prog_str):
    prog_ast = ast.parse(prog_str)
    fn = prog_ast.body[-1]

    mapping = {}

    # Rename the arguments of the function
    for arg_idx, arg in enumerate(fn.args.args):
        new_arg_id = "arg" + str(arg_idx)
        mapping[arg.arg] = new_arg_id
        fn.args.args[arg_idx].arg = new_arg_id
    n_args = len(fn.args.args)

    def remap_id(id):
        if id not in mapping:
            mapping[id] = "v" + str(len(mapping) - n_args)
        return mapping[id]

    # Rename the variables of the function
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign):
            assert len(stmt.targets) == 1
            target = stmt.targets[0]

            if isinstance(target, ast.Name):
                target.id = remap_id(target.id)
            else:
                print("ERROR: Unsupported target type %s" % type(target))
                assert False

            if isinstance(stmt.value, ast.Call):
                for arg in stmt.value.args:
                    if isinstance(arg, ast.Name):
                        arg.id = remap_id(arg.id)
            elif isinstance(stmt.value, ast.Name):
                stmt.value.id = remap_id(stmt.value.id)
            elif isinstance(stmt.value, ast.Constant):
                pass
            elif isinstance(stmt.value, ast.BinOp):
                if isinstance(stmt.value.left, ast.Name):
                    stmt.value.left.id = remap_id(stmt.value.left.id)
                if isinstance(stmt.value.right, ast.Name):
                    stmt.value.right.id = remap_id(stmt.value.right.id)
            else:
                print("ERROR: Unsupported value type %s" % type(stmt.value))
                assert False

        if isinstance(stmt, ast.Return):
            if isinstance(stmt.value, ast.Name):
                stmt.value.id = remap_id(stmt.value.id)
            elif isinstance(stmt.value, ast.Call):
                for arg in stmt.value.args:
                    if isinstance(arg, ast.Name):
                        arg.id = remap_id(arg.id)
            elif isinstance(stmt.value, ast.Constant):
                pass
            else:
                print("ERROR: Unsupported return value type %s" % type(stmt.value))
                assert False

    return ast.unparse(fn)


class NumpyTarget(Target):
    def __init__(self) -> None:
        return

    def get_candidates_for_arg_shapes(
        self, arg_shapes, max_depth, ops=[], constants=[]
    ):
        # Map op names to ops
        print(ops)

        op_name_to_op = {op_tuple[0].__name__: op_tuple for op_tuple in NUMPY_GRAMMAR}
        operations = [op_name_to_op[op_name.split(".")[1]] for op_name in ops]

        if len(operations) == 0:
            operations = NUMPY_GRAMMAR

        candidates = enumerate_candidates(arg_shapes, operations, constants, max_depth)

        return candidates

    def get_stubs_for_arg_shapes(self, arg_shapes, max_depth, ops=[], constants=[]):
        candidates = self.get_candidates_for_arg_shapes(
            arg_shapes, max_depth, ops, constants
        )

        candidates_w_rhs_add = extend_candidates_w_rhs_add(candidates)

        candidates_w_rhs_add = compile_candidates(
            candidates_w_rhs_add, desc="Creating templates"
        )
        candidates += candidates_w_rhs_add

        return [Stub(c.ir, c.fn) for c in candidates]

    def get_stubs(self, input_module, max_depth, ops, constants):
        # Parse arg shapes
        with Context(), Location.unknown():
            mlir_synth.register_dialects()
            mlir_synth.register_passes()
            m = Module.parse(input_module)
            arg_shapes = get_arg_shapes(m)

        return self.get_stubs_for_arg_shapes(arg_shapes, max_depth, ops, constants)

    def construct_function_ast(
        self, sol_sketch, source_function, func_idx, program_name
    ):
        mapping = {}

        def get_mapped_name(obj, name):
            map_id = (obj, name)

            if map_id not in mapping:
                mapping[map_id] = ast.Name("x%d" % len(mapping.keys()))

            return mapping[map_id]

        def get_mapped_name_existing(obj, name):
            map_id = (obj, name)
            if map_id not in mapping:
                raise Exception("Name " + name + " not found in mapping")
            return mapping[map_id]

        def build_body(sol_sk, new_body):
            results = []
            for operand in sol_sk.operands:
                results.append(build_body(operand, new_body))

            fn_str = sol_sk.sketch.stub.original
            fn_ast = ast.parse(fn_str)
            fn_body = fn_ast.body[0].body

            last_mapped_name = None
            for stmt in fn_body:
                # Map names of defs
                if isinstance(stmt, ast.Return):
                    mapped_name = get_mapped_name(sol_sk, stmt)
                else:
                    assert len(stmt.targets) == 1
                    target = stmt.targets[0]
                    mapped_name = get_mapped_name(sol_sk, target.id)

                # Clone the statement
                if isinstance(stmt, ast.Return):
                    new_stmt = ast.Assign([mapped_name], stmt.value, lineno=stmt.lineno)
                else:
                    new_stmt = copy.deepcopy(stmt)

                # Replace the names
                # - In LHS
                if isinstance(stmt, ast.Return):
                    pass
                else:
                    new_stmt.targets[0] = get_mapped_name(
                        sol_sk, new_stmt.targets[0].id
                    )

                # - In RHS
                if isinstance(new_stmt.value, ast.Call):
                    for arg_idx, arg in enumerate(new_stmt.value.args):
                        if isinstance(arg, ast.Name):
                            # Variables
                            if arg.id.startswith("v"):
                                new_stmt.value.args[arg_idx] = get_mapped_name_existing(
                                    sol_sk, arg.id
                                )

                            # Arguments
                            elif arg.id.startswith("arg"):
                                arg_id = int(arg.id[3:])

                                # If argument is result of a previous sketch (aka unknown),
                                # replace with the return name
                                if arg_id in sol_sk.sketch.unknown_idxs:
                                    unknown_idx = sol_sk.sketch.unknown_idxs.index(
                                        arg_id
                                    )
                                    new_stmt.value.args[arg_idx] = results[unknown_idx]
                                else:
                                    arg_id_orig = sol_sk.sketch.arg_idxs[arg_id]
                                    new_stmt.value.args[arg_idx] = ast.Name(
                                        "arg%d" % arg_id_orig
                                    )
                    last_mapped_name = mapped_name
                elif isinstance(new_stmt.value, ast.Constant):
                    new_stmt.value = ast.Constant(new_stmt.value.value)
                    last_mapped_name = mapped_name

                elif isinstance(new_stmt.value, ast.Name):
                    # Skip simple assignments
                    if new_stmt.value.id.startswith(
                        "x"
                    ) or new_stmt.value.id.startswith("v"):
                        continue

                    arg_id = int(new_stmt.value.id[3:])
                    arg_id_orig = sol_sk.sketch.arg_idxs[arg_id]
                    new_stmt.value = ast.Name("arg%d" % arg_id_orig)

                    last_mapped_name = mapped_name
                else:
                    raise Exception("Unsupported type " + str(type(new_stmt.value)))

                # Append the statement to the new body
                new_body.append(new_stmt)

            return last_mapped_name

        new_body = []
        last_result = build_body(sol_sketch, new_body)
        new_body.append(ast.Return(last_result))

        # Create new function
        arg_names = [
            ast.arg("arg%d" % i, None) for i in range(len(source_function.arguments))
        ]
        args = ast.arguments(
            args=arg_names,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
            posonlyargs=[],
        )
        new_fn = ast.FunctionDef(
            name="fn_%d" % func_idx,
            args=args,
            decorator_list=[],
            lineno=0,
            body=new_body,
        )

        # Print the new body
        function_py = ast.unparse(new_fn)
        self.function_pys.append(function_py)

        return function_py

    def initialize_ast(self, module):
        self.main_function_py = replace_return_value(
            get_last_function(affine_to_python(str(module)))
        )
        self.changed_sizes = get_sizes(module)

        self.function_pys = []

    def construct_program_ast(self):
        prog_str = "\n".join(self.function_pys) + "\n" + self.main_function_py

        prog_str = inline_ast_functions(prog_str)
        prog_str = restore_sizes(prog_str, self.changed_sizes)
        prog_str = inline_stmts(prog_str)
        prog_str = normalize_identifiers(prog_str)

        prog_str = prog_str.replace("jnp.", "np.")

        return prog_str
