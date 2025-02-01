from mlir_synth.ir import *
from mlir_synth.passmanager import *

from synthesis import *
from target.target import Target
from mlir_helpers import *


OPERATIONS = [
    "chlo.broadcast_divide",
    "chlo.broadcast_add",
    "chlo.broadcast_subtract",
    "chlo.broadcast_multiply",
    "stablehlo.dot",
    "stablehlo.slice",
    "stablehlo.reduce",
    "stablehlo.dot_general",
    "stablehlo.transpose",
    "stablehlo.select",
    "stablehlo.sqrt",
    "stablehlo.compare",
    "chlo.broadcast_power",
]


def extend_w_rhs_add(ir_str):
    with Context(), Location.unknown():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        ir = Module.parse(ir_str)

        func = get_function(ir)
        block = func.regions[0].blocks[0]

        returnOp = list(block.operations)[-1]
        returnOperand = returnOp.operands[0]

        # Skip functions that have only a single operation, which is a return
        if len(block.operations) == 1:
            return None

        # Update the function signature to include a new argument with the
        # same type as the return type
        arg_types = func.type.inputs + [returnOperand.type]
        return_type = returnOperand.type

        func_wip = FuncOp(
            name="func_wip",
            type=FunctionType.get(inputs=arg_types, results=[return_type]),
        )

        func_wip.add_entry_block()
        block_wip = func_wip.body.blocks[0]
        ip_wip = InsertionPoint(block_wip)

        value_to_wip_value = {}
        for arg_idx, arg in enumerate(block.arguments):
            value_to_wip_value[arg] = block_wip.arguments[arg_idx]

        # Add operations from the original function, but leave out the return
        op_wip = None
        for op in list(block.operations):
            op = op.operation

            if op.name == "func.return":
                continue

            # Create the operation
            op_wip = Operation.create(
                name=op.name,
                operands=[value_to_wip_value[o] for o in op.operands],
                attributes={a.name: a.attr for a in op.attributes},
                results=[r.type for r in op.results],
                regions=len(op.regions),
            )

            # Clone regions
            for region_idx, region in enumerate(op.regions):
                region_wip = op_wip.regions[region_idx]

                for block in region.blocks:
                    region_block_wip = region_wip.blocks.append(
                        *[x.type for x in block.arguments]
                    )
                    ip_region_block = InsertionPoint(region_block_wip)

                    for arg_idx, arg in enumerate(block.arguments):
                        value_to_wip_value[arg] = region_block_wip.arguments[arg_idx]

                    # Add operations from block to block_wip
                    for block_op in block.operations:
                        block_op = block_op.operation
                        block_op_wip = Operation.create(
                            name=block_op.name,
                            operands=[value_to_wip_value[o] for o in block_op.operands],
                            attributes={a.name: a.attr for a in block_op.attributes},
                            results=[r.type for r in block_op.results],
                        )
                        ip_region_block.insert(block_op_wip)

                        for res_idx, res in enumerate(block_op.results):
                            value_to_wip_value[res] = block_op_wip.results[res_idx]

            ip_wip.insert(op_wip)

            # Update the value map
            for res_idx, res in enumerate(op.results):
                value_to_wip_value[res] = op_wip.results[res_idx]

        # Create an mhlo.add operation that adds op_wip and the last argument
        # of the function
        add_op_wip = Operation.create(
            name="mhlo.add",
            operands=[op_wip.results[0], block_wip.arguments[-1]],
            attributes={},
            results=[return_type],
        )
        ip_wip.insert(add_op_wip)

        # Create a return operation
        return_op_wip = Operation.create(
            name="func.return", operands=[add_op_wip.results[0]], attributes={}
        )
        ip_wip.insert(return_op_wip)

        return str(func_wip)


class HloTarget(Target):
    def __init__(self) -> None:
        self.functions = []
        self.main_function = None
        return

    def get_stubs(self, input_module, max_depth, ops, constants):
        with Context(), Location.unknown():
            mlir_synth.register_dialects()

            operations = ops if len(ops) > 0 else OPERATIONS
            print(operations)

            options = {
                "maxNumOps": 1,
                "printValidCandidates": False,
                "printInvalidCandidates": False,
                "printSynthesisSteps": False,
                "stopOnSolutionCandidate": False,
                "skipMergeCandidateArguments": True,
                "ignoreEquivalentCandidates": False,
            }
            irs = []

            m = Module.parse(input_module)
            arg_types = list(set(get_arg_types(m)))
            return_type = get_return_type(m)

            mlir_synth.enumerate_one_op(
                operations, arg_types, return_type, options, irs
            )

        irs = dedup_irs(irs)
        irs_w_rhs_add = [extend_w_rhs_add(ir) for ir in irs]
        irs += dedup_irs([ir for ir in irs_w_rhs_add if ir is not None])

        return [Stub(ir, ir) for ir in irs]

    def construct_function_ast(
        self, sol_sketch, source_function, func_idx, program_name
    ):
        with Context(), Location.unknown():
            mlir_synth.register_dialects()

            mod = Module.parse("module { %s }" % source_function)
            func = get_function(mod)

            # Function
            # - Argument types
            arg_types = [get_as_tensor_type(arg.type) for arg in func.arguments]

            # - Return type
            return_op = func.body.blocks[0].operations[
                len(func.body.blocks[0].operations) - 1
            ]
            return_type = get_as_tensor_type(return_op.operands[0].type)

            # - Function op
            func_wip = FuncOp(
                name="raised",
                type=FunctionType.get(inputs=arg_types, results=[return_type]),
            )
            func_wip.add_entry_block()
            block_wip = func_wip.body.blocks[0]
            ip_wip = InsertionPoint(block_wip)

            def add_op(sol_sketch):
                parents = [add_op(operand) for operand in sol_sketch.operands]

                mod = Module.parse(sol_sketch.sketch.stub.ir)
                func = get_function(mod)

                value_to_wip_value = {}
                # Add arguments
                for arg_idx_idx, arg_idx in enumerate(sol_sketch.sketch.arg_idxs):
                    value_to_wip_value[func.arguments[arg_idx_idx]] = (
                        func_wip.arguments[arg_idx]
                    )
                # Add unknowns
                for unknown_idx, parent in zip(sol_sketch.sketch.unknown_idxs, parents):
                    value_to_wip_value[func.arguments[unknown_idx]] = parent

                last_result = None
                for op in func.body.blocks[0].operations:
                    op = op.operation
                    if op.name == "func.return":
                        if not last_result:
                            return value_to_wip_value[op.operands[0]]
                        continue

                    operands = [value_to_wip_value[operand] for operand in op.operands]

                    new_op = Operation.create(
                        name=op.name,
                        operands=operands,
                        attributes={a.name: a.attr for a in op.attributes},
                        results=[r.type for r in op.results],
                    )
                    last_result = new_op.result
                    ip_wip.insert(new_op)

                    # Update the value map
                    for res_idx, res in enumerate(op.results):
                        value_to_wip_value[res] = new_op.results[res_idx]

                return last_result

            last_result = add_op(sol_sketch)

            # Add return
            ip_wip.insert(ReturnOp([last_result]))

        self.functions.append(func_wip)

        return func_wip

    def initialize_ast(self, module):
        # Get the last function in the module. This is the function that calls
        # the synthesized functions.
        last_func = None
        for maybeFunc in module.body.operations:
            if isinstance(maybeFunc, FuncOp):
                last_func = maybeFunc
        assert last_func is not None
        self.main_function = str(last_func).replace("memref", "tensor")

    def construct_program_ast(self):
        new_module_str = ""

        for function_idx, function in enumerate(self.functions):
            new_module_str += str(function).replace("raised", "fn_%d" % function_idx)
        new_module_str += self.main_function

        new_module_str = new_module_str.replace("\n", "\n  ")
        new_module_str = "module {\n  " + new_module_str + "\n}"

        return new_module_str
