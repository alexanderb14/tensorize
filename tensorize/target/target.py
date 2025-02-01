class Target:
    def get_stubs(self, input_module, max_depth, ops, constants):
        raise NotImplementedError

    def initialize_ast(self, module):
        pass

    def construct_function_ast(
        self, sol_sketch, source_function, func_idx, program_name
    ):
        raise NotImplementedError

    def construct_program_ast(self):
        raise NotImplementedError
