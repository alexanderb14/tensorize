import copy
import itertools
import logging
import stopit
from collections import ChainMap
from enum import Enum

import sympy
import sympy.core
from sympy.core.numbers import *
from sympy.core.relational import *
from sympy.parsing.sympy_parser import parse_expr


stopit_logger = logging.getLogger("stopit")
stopit_logger.setLevel(logging.ERROR)


class ArgProperty(Enum):
    Regular = 1
    Symmetric = 2
    LowerTriangular = 3


class Done:
    def __init__(self):
        pass

    def __repr__(self) -> str:
        return "Done"
        # return "\033[1m" + "Done" + "\033[0m"


def print_green(s):
    print("\033[1;42m" + str(s) + "\033[0m")


def get_unknown_basenames_nd(syms):
    def helper(syms):
        if isinstance(syms, list):
            return list(itertools.chain(*[helper(s) for s in syms]))
        return [s.name for s in syms.free_symbols if str(s).startswith("X")]

    unknown_names = helper(syms)
    return list(set([name.split("_")[0] for name in unknown_names]))


def get_symbols_nd(name, shape, arg_property=ArgProperty.Regular):
    def helper(name, shape, is_first_call=False):
        if len(shape) == 0:
            return sympy.symbols(name)

        if is_first_call:
            return [helper(name + "_" + str(i), shape[1:]) for i in range(shape[0])]
        else:
            return [helper(name + "x" + str(i), shape[1:]) for i in range(shape[0])]

    def symmetric_helper(name, shape):
        if len(shape) == 0:
            return sympy.symbols(name)
        elif len(shape) == 1:
            syms = []
            for i in range(shape[0]):
                syms.append(sympy.symbols(name + "_" + str(i)))
            return syms
        elif len(shape) == 2:
            syms = []
            for i in range(shape[0]):
                row = []
                for j in range(shape[1]):
                    if i <= j:
                        row.append(sympy.symbols(name + "_" + str(i) + "x" + str(j)))
                    else:
                        row.append(sympy.symbols(name + "_" + str(j) + "x" + str(i)))
                syms.append(row)
            return syms
        else:
            raise Exception("Unsupported shape: " + str(shape))

    def lower_triangular_helper(name, shape):
        if len(shape) == 0:
            return sympy.symbols(name)
        elif len(shape) == 1:
            syms = []
            for i in range(shape[0]):
                syms.append(sympy.symbols(name + "_" + str(i)))
            return syms
        elif len(shape) == 2:
            syms = []
            for i in range(shape[0]):
                row = []
                for j in range(shape[1]):
                    if i > j:
                        row.append(sympy.symbols(name + "_" + str(i) + "x" + str(j)))
                    elif i == j:
                        row.append(One())
                    else:
                        row.append(Zero())
                syms.append(row)
            return syms
        else:
            raise Exception("Unsupported shape: " + str(shape))

    if arg_property == ArgProperty.Regular:
        return helper(name, shape, True)
    elif arg_property == ArgProperty.Symmetric:
        return symmetric_helper(name, shape)
    elif arg_property == ArgProperty.LowerTriangular:
        syms = lower_triangular_helper(name, shape)
        return syms
    else:
        raise Exception("Unsupported arg_property: " + str(arg_property))


def get_shape_nd(syms):
    def helper(syms):
        if isinstance(syms, list):
            return [len(syms)] + helper(syms[0])
        else:
            return []

    return helper(syms)


def substract_symbols_nd(syms1, syms2):
    def helper(syms1, syms2):
        if isinstance(syms1, list):
            return [helper(s1, s2) for s1, s2 in zip(syms1, syms2)]

        # Check if float and substract. Do this now because the subsequent checks will
        # fail for floats
        if isinstance(syms1, float) or isinstance(syms2, float):
            return syms1 - syms2

        # Cannot substract relationals, such as Eq, Lt, Gt, etc.
        if syms1.is_Relational or syms2.is_Relational:
            return None

        return syms1 - syms2

    return helper(syms1, syms2)


def is_equal_nd(syms1, syms2, ignore_done=False):
    if get_shape_nd(syms1) != get_shape_nd(syms2):
        return False

    def helper(syms1, syms2):
        if isinstance(syms1, list):
            return all([helper(s1, s2) for s1, s2 in zip(syms1, syms2)])

        if ignore_done and (isinstance(syms1, Done) or isinstance(syms2, Done)):
            return True

        # First check if they are string equal. This is cheaper than the sympy equality check
        if str(syms1) == str(syms2):
            return True

        return syms1 == syms2

    return helper(syms1, syms2)


def get_unknowns_nd(syms):
    def helper(syms):
        if isinstance(syms, list):
            return [helper(s) for s in syms]
        return set([s for s in syms.free_symbols if str(s).startswith("X")])

    return helper(syms)


def collect_unknowns_nd(syms):
    def helper(syms):
        if isinstance(syms, list) or isinstance(syms, set):
            return list(itertools.chain(*[helper(s) for s in syms]))

        if isinstance(syms, dict):
            return list(itertools.chain([s for s in syms.keys()]))

        if isinstance(syms, Done):
            return []

        return [str(s) for s in syms.free_symbols if str(s).startswith("X")]

    return list(set(helper(syms)))


def mask_away_done_nd(syms, mask):
    def helper(syms, mask):
        if isinstance(syms, list):
            return [helper(s, m) for s, m in zip(syms, mask)]

        if isinstance(mask, Done):
            return {}
        else:
            return syms

    return helper(syms, mask)


def has_unknowns_nd(syms):
    def helper(syms):
        if isinstance(syms, list):
            return any([helper(s) for s in syms])

        if isinstance(syms, sympy.Symbol) or isinstance(syms, sympy.core.Basic):
            return any([str(s).startswith("X") for s in syms.free_symbols])
        else:
            return False

    return helper(syms)


def is_any_intersection_nonempty_nd(syms1, syms2):
    def helper(syms1, syms2):
        if isinstance(syms1, list):
            return any([helper(s1, s2) for s1, s2 in zip(syms1, syms2)])

        if isinstance(syms1, float) or isinstance(syms2, float):
            return True
        if isinstance(syms1, Float) or isinstance(syms2, Float):
            return True

        if isinstance(syms1, Done) or isinstance(syms2, Done):
            return True

        return len(syms1.free_symbols.intersection(syms2.free_symbols)) != 0

    return helper(syms1, syms2)


def has_nones_in_nd(syms):
    def helper(syms):
        if isinstance(syms, dict):
            return any([helper(s) for s in syms.values()])

        if isinstance(syms, list):
            return any([helper(s) for s in syms])
        return syms == None

    return helper(syms)


def get_num_elements_nd(syms):
    def helper(syms):
        if isinstance(syms, list):
            return sum([helper(s) for s in syms])

        if isinstance(syms, Done):
            return 0

        return len(syms)

    return helper(syms)


def has_all_zeros_in_nd(syms):
    def helper(syms):
        if isinstance(syms, list):
            return all([helper(s) for s in syms])

        if isinstance(syms, Done):
            return 0

        return isinstance(syms, Zero)

    return helper(syms)


def has_all_true_nd(eles):
    def helper(eles):
        if isinstance(eles, list):
            return all([helper(e) for e in eles])
        return eles == True

    return helper(eles)


def has_duplicate_keys_nd(sol_dict):
    all_keys = {}

    def helper(sol_dict):
        if isinstance(sol_dict, list):
            return any([helper(s) for s in sol_dict])
        if isinstance(sol_dict, dict):
            for k in sol_dict.keys():
                if k in all_keys:
                    return True
                all_keys[k] = True
        return False

    return helper(sol_dict)


def reparse_without_eval_nd(syms):
    def helper(syms):
        if isinstance(syms, list):
            return [helper(s) for s in syms]
        return parse_expr(str(syms), evaluate=False)

    return helper(syms)


def project_solution_to_input_nd(solution_dicts_nd):
    # Get the unknown names
    def get_unknown_basenames_nd(solution_dicts_nd):
        def helper(sol_dict_or_list):
            if isinstance(sol_dict_or_list, list):
                return list(itertools.chain(*[helper(s) for s in sol_dict_or_list]))
            if isinstance(sol_dict_or_list, dict):
                return [k for k in sol_dict_or_list.keys() if k.startswith("X")]
            return []

        unknown_names = helper(solution_dicts_nd)
        return sorted(list(set([name.split("_")[0] for name in unknown_names])))

    unknown_names = get_unknown_basenames_nd(solution_dicts_nd)

    # Project each unknown into a separate solution_nd
    solution_nds = {}
    for unknown_name in unknown_names:
        # Get the maximum indices for each dimension of the solution_nd
        max_indices = []

        def get_max_indices_nd(solution_dicts_nd):
            def helper(sol_dict_or_list):
                if isinstance(sol_dict_or_list, list):
                    for s in sol_dict_or_list:
                        helper(s)
                elif isinstance(sol_dict_or_list, dict):
                    for unk_sym, sol in sol_dict_or_list.items():
                        if "_" in unk_sym:
                            if unknown_name != unk_sym.split("_")[0]:
                                continue
                            if "_" not in unk_sym and unknown_name != unk_sym:
                                continue
                            indices_str = unk_sym.split("_")[1]

                            indices = (
                                [int(c) for c in indices_str.split("x")]
                                if len(indices_str) > 0
                                else []
                            )

                            for i, idx in enumerate(indices):
                                if len(max_indices) <= i:
                                    max_indices.append(idx)
                                else:
                                    max_indices[i] = max(max_indices[i], idx)

            return helper(solution_dicts_nd)

        get_max_indices_nd(solution_dicts_nd)

        # Create a solution_nd of Nones, with the same shape as the solution_shape
        solution_shape = [i + 1 for i in max_indices]
        solution_nd = Done()
        for dim in reversed(solution_shape):
            if isinstance(solution_nd, Done):
                solution_nd = [Done() for _ in range(dim)]
            else:
                solution_nd = [copy.deepcopy(solution_nd) for _ in range(dim)]

        # Project the solution into the solution_nd
        def project_nd(sol_dict_or_list, sol_nd):
            if isinstance(sol_dict_or_list, list):
                for s in sol_dict_or_list:
                    project_nd(s, sol_nd)
            elif isinstance(sol_dict_or_list, dict):
                for unk_sym, sol in sol_dict_or_list.items():
                    if unknown_name != unk_sym.split("_")[0]:
                        continue
                    if "_" not in unk_sym and unknown_name != unk_sym:
                        continue

                    indices_str = unk_sym.split("_")[1] if "_" in unk_sym else ""
                    indices = (
                        [int(c) for c in indices_str.split("x")]
                        if len(indices_str) > 0
                        else []
                    )

                    if len(indices) == 0:
                        sol_nd = sol
                    elif len(indices) == 1:
                        sol_nd[indices[0]] = sol
                    elif len(indices) == 2:
                        sol_nd[indices[0]][indices[1]] = sol
                    elif len(indices) == 3:
                        sol_nd[indices[0]][indices[1]][indices[2]] = sol
                    else:
                        raise Exception("Unsupported shape")

            return sol_nd

        solution_nds[unknown_name] = project_nd(solution_dicts_nd, solution_nd)

    return solution_nds


def pattern_match_nd(symsSpec, symsSketch):
    def canonicalize_mul(expr):
        """
        Canonicalize mul, such that the pow is on the left hand side.
        This is useful because sympy represents a div with a mul and
        the order is important for pattern matching."""

        def helper(expr_):
            if isinstance(expr_, list):
                return [helper(l) for l in expr_]

            elif isinstance(expr_, sympy.Mul):
                if any([isinstance(s, sympy.core.power.Pow) for s in expr_.args]):
                    expr_pow = [
                        s for s in expr_.args if isinstance(s, sympy.core.power.Pow)
                    ]
                    expr_other = [
                        s for s in expr_.args if not isinstance(s, sympy.core.power.Pow)
                    ]
                    if not (len(expr_pow) == 1 and len(expr_other) == 1):
                        return expr_

                    return sympy.Mul(expr_pow[0], expr_other[0], evaluate=False)
            else:
                return expr_

        return helper(expr)

    def extract_dicts_from_expr(spec, sketch):
        sol_dict = {}

        def helper(eleSpec, eleSketch):
            if isinstance(eleSketch, sympy.Symbol):
                if str(eleSketch).startswith("X"):
                    unknown_name = str(eleSketch)
                    sol_dict[unknown_name] = eleSpec
                    return True
                if str(eleSketch) == str(eleSpec):
                    return True
                return False

            if type(eleSpec) != type(eleSketch):
                return False
            if len(eleSpec.args) != len(eleSketch.args):
                return False

            if isinstance(eleSpec, sympy.Float) and isinstance(eleSketch, sympy.Float):
                if eleSpec != eleSketch:
                    return False

            return all(
                [
                    helper(x, y)
                    for x, y in zip(
                        eleSpec.args,
                        eleSketch.args,
                    )
                ]
            )

        ok = helper(spec, sketch)
        return ok, sol_dict

    def extract_sol_dict(symsSpec, symsSketch):
        if isinstance(symsSpec, list):
            return [extract_sol_dict(s1, s2) for s1, s2 in zip(symsSpec, symsSketch)]

        if type(symsSpec) == type(symsSketch) and (
            isinstance(symsSpec, sympy.Expr) or isinstance(symsSpec, Relational)
        ):
            # print("!!!Match!!! ", type(symsSpec), type(symsSketch))
            ok, sol_dicts = extract_dicts_from_expr(symsSpec, symsSketch)
            # print(ok)
            return ok, sol_dicts

    def select_element(sol_dicts, name):
        def helper(sol_dict):
            if isinstance(sol_dict, list):
                return [helper(s) for s in sol_dict]
            return sol_dict[name]

        return helper(sol_dicts)

    def split_sol_dicts(sol_dicts, idx):
        if isinstance(sol_dicts, list):
            return [split_sol_dicts(s, idx) for s in sol_dicts]
        return sol_dicts[idx]

    symsSpec = canonicalize_mul(symsSpec)
    symsSketch = canonicalize_mul(symsSketch)

    sol_dicts = extract_sol_dict(symsSpec, symsSketch)
    if has_nones_in_nd(sol_dicts) or get_num_elements_nd(sol_dicts) == 0:
        return None

    ok, sol_dict = split_sol_dicts(sol_dicts, 0), split_sol_dicts(sol_dicts, 1)
    if not has_all_true_nd(ok):
        return None

    # Duplicate keys mean that the same unknown is used in multiple places
    if has_duplicate_keys_nd(sol_dict):
        return None

    if get_num_elements_nd(sol_dict) == 0:
        return None

    sols = project_solution_to_input_nd(sol_dict)

    if has_nones_in_nd(sols):
        return None

    return sols


def solve_nd(eq, unknowns, solution_shape):
    # Create a solution_nd of Nones, with the same shape as the solution_shape
    solution_nd = None
    for dim in reversed(solution_shape):
        if solution_nd == None:
            solution_nd = [None for _ in range(dim)]
        else:
            solution_nd = [copy.deepcopy(solution_nd) for _ in range(dim)]

    def solve(eq, unknowns, solution_nd):
        if isinstance(eq, list):
            return [solve(e, u, solution_nd) for e, u in zip(eq, unknowns)]

        # Check if the equation contains a done. In this case, we don't need to
        # consider it anymore
        if "Done" in str(eq):
            return Done()

        # Check if the solution is valid
        # - If the substraction result is 0, and there were no unknowns, then the solution is valid
        if eq == 0 and len(unknowns) == 0:
            return Done()

        try:
            with stopit.ThreadingTimeout(1, swallow_exc=False) as timeout_ctx:
                solutions = sympy.solvers.solve(eq, list(unknowns), dict=True)
        except stopit.utils.TimeoutException as e:
            # print("Warning: Timeout exceeded in sympy.solvers.solve")
            solutions = []
            pass
        except (NotImplementedError, TypeError, KeyError) as e:
            # print("Warning: Exception in sympy.solvers.solve: " + str(e))
            solutions = []
            pass

        except AttributeError:
            # print("Warning: Encountered AttributeError while solving")
            solutions = []
            pass

        # Check if the solution is valid
        for solution in solutions:
            # Check if substitution is valid
            if sympy.simplify(eq.subs(solution)) != 0:
                solutions.remove(solution)

        solutions = dict(ChainMap(*solutions))
        solutions = {str(k): v for k, v in solutions.items()}

        # - If there are no solutions and the equation is non zero, then the solution is invalid
        if eq != 0 and len(solutions) == 0:
            return None

        return solutions

    sol_dict = solve(eq, unknowns, solution_nd)

    # Duplicate keys mean that the same unknown is used in multiple places
    if has_duplicate_keys_nd(sol_dict):
        return None

    # Check if the solution is valid
    if has_nones_in_nd(sol_dict) or get_num_elements_nd(sol_dict) == 0:
        return None

    # Check if the solution is complete
    unks_eq = set(collect_unknowns_nd(mask_away_done_nd(unknowns, sol_dict)))
    unks_sol = set(collect_unknowns_nd(sol_dict))
    if unks_eq != unks_sol:
        return None

    sols = project_solution_to_input_nd(sol_dict)
    if len(sols.keys()) == 0:
        return None

    return sols.values()


def apply_nd(syms, func=sympy.simplify, timeout=1):
    def helper(syms):
        if isinstance(syms, list):
            return [helper(s) for s in syms]
        return func(syms)

    try:
        with stopit.ThreadingTimeout(timeout, swallow_exc=False):
            return helper(syms), "ok"
    except stopit.utils.TimeoutException as e:
        print("Warning: Timeout in apply_nd for function: " + str(func.__name__))
        return None, "timeout_" + str(func.__name__)


def get_unique_free_symbols_nd(syms_in):
    free_symbols = set()
    syms_in_str = str(syms_in)

    def helper(syms):
        if isinstance(syms, float) or isinstance(syms, bool) or isinstance(syms, int):
            free_symbols.add(syms)
        elif isinstance(syms, sympy.core.Basic):
            for sym in syms.free_symbols:
                free_symbols.add(sym)

            for ty in [sympy.core.numbers.Integer, sympy.core.numbers.Float]:
                for const in syms.atoms(ty):
                    # Check if it is in the string representation of the input, as sympy.atoms()
                    # returns constants that are not in the expression itself.
                    # Example: -1 for A - B, because A - B is represented as Add(A, Mul(-1, B))
                    is_in_str = str(const) in syms_in_str

                    # Only add the constant if it is in the expression itself
                    if is_in_str:
                        free_symbols.add(const)
        elif isinstance(syms, list):
            for sym in syms:
                helper(sym)
        elif isinstance(syms, Done):
            pass
        elif syms == None:
            pass
        else:
            raise Exception("Unsupported type: " + str(type(syms)))

    helper(syms_in)
    return free_symbols


def get_num_unique_free_symbols_nd(syms):
    return len(get_unique_free_symbols_nd(syms))


def get_simp_score_nd(inputs):
    simp_score = 0

    for input in inputs:
        num_free_syms = len(get_unique_free_symbols_nd(input))
        density = get_undone_to_done_ratio(input)

        simp_score += num_free_syms * density

    return simp_score


def get_constants_nd(syms):
    constants = set()

    def helper(syms):
        if isinstance(syms, sympy.core.Basic):
            for ty in [sympy.core.numbers.Integer, sympy.core.numbers.Float]:
                for const in syms.atoms(ty):
                    constants.add(const)

        elif isinstance(syms, list):
            for sym in syms:
                helper(sym)

    helper(syms)
    return constants


def get_max_ops_count_nd(syms):
    def helper(syms):
        if isinstance(syms, list):
            return max([helper(s) for s in syms])
        if isinstance(syms, Done):
            return 0

        return Expr.count_ops(syms)

    return helper(syms)


def print_symbols_nd(syms):
    def helper(syms, indent):
        if isinstance(syms, list):
            for sym in syms:
                helper(sym, indent)
            print()
        else:
            print(str(syms), end="")
            print(", ", end="")

    helper(syms, 0)
    print()


def get_undone_to_done_ratio(syms):
    # Get the number of elements
    def ele_helper(syms):
        if isinstance(syms, list):
            return sum([ele_helper(s) for s in syms])
        return 1

    num_elements = ele_helper(syms)

    # Get the number of Done's
    def done_helper(syms):
        if isinstance(syms, list):
            return sum([done_helper(s) for s in syms])
        if isinstance(syms, Done):
            return 1
        return 0

    num_dones = done_helper(syms)

    return (num_elements - num_dones) / num_elements


def copy_arg_symbols(arg_symbols):
    def helper(syms):
        if isinstance(syms, list):
            return [helper(s) for s in syms]

        if isinstance(syms, Zero):
            return Zero()
        if isinstance(syms, One):
            return One()

        return sympy.Symbol(syms.name)

    return helper(arg_symbols)
