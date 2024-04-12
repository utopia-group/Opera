import os
import tempfile
from collections import defaultdict
from subprocess import CalledProcessError, run
from typing import Optional

from pycparser import CParser, c_ast, c_generator

from b2s.input_config import SynthesisFacts


def run_sketch_solver(sketch_code: str, options: list[str] = []) -> Optional[str]:
    """Run the sketch solver on the given sketch code.

    Args:
        sketch_code: The sketch code to run.
        options: Additional options to pass to the sketch solver.

    Returns:
        The output of the sketch solver.
    """
    # todo: remove the debug print
    with open("tmp.sk", "w") as file:
        file.write(sketch_code)

    with tempfile.NamedTemporaryFile(mode="w") as temp_file:
        temp_name = temp_file.name + ".sk"

    with open(temp_name, "w") as file:
        file.write(sketch_code)

    try:
        sketch_output = run(
            [
                "sketch",
                "--fe-output-dir",
                tempfile.gettempdir() + "/",
                *options,
                temp_name,
            ],
            capture_output=True,
            encoding="utf-8",
            check=True,
        )

        return sketch_output.stdout
    finally:
        try:
            os.remove(temp_name)
        except FileNotFoundError:
            pass


def run_sketch_solver_as_c(sketch_code: str, options: list[str] = []) -> Optional[str]:
    """Run the sketch solver on the given sketch code with C code as output.

    Args:
        sketch_code: The sketch code to run.
        options: Additional options to pass to the sketch solver.

    Returns:
        The output of the sketch solver.
    """
    with tempfile.NamedTemporaryFile(mode="w") as file:
        tmp_name = file.name

    tmp_src = tmp_name + ".cpp"
    tmp_h = tmp_name + ".h"

    try:
        stdout = run_sketch_solver(
            sketch_code,
            [
                "--fe-output-code",
                "--fe-output-prog-name",
                os.path.basename(tmp_name),
                *options,
            ],
        )
        print(stdout)

        with open(tmp_src, "r") as file:
            return file.read()
    except CalledProcessError as e:
        print("sketch failed with error code", e.returncode)
        print(e.stdout)
        print(e.stderr)
        return None
    finally:
        try:
            os.remove(tmp_src)
            os.remove(tmp_h)
        except FileNotFoundError:
            pass


class FuncDefVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.assignment_map: defaultdict[str, dict[str, str]] = defaultdict(dict)
        self.result_map: dict[str, str] = {}
        self.generator = c_generator.CGenerator()

    def visit_FuncDef(self, node):
        func_name = node.decl.name
        for stmt in node.body.block_items:
            match stmt:
                case c_ast.Assignment(op=op, lvalue=lvalue, rvalue=rvalue):
                    assert op == "=", "Only support = assignments??"
                    lval_str = self.generator.visit(lvalue)
                    rval_str = self.generator.visit(rvalue)
                    match lval_str:
                        case "_out":
                            self.result_map[func_name] = self.assignment_map[
                                func_name
                            ].get(rval_str, rval_str)
                        case _:
                            self.assignment_map[func_name][lval_str] = rval_str
                case c_ast.Decl(name=name, init=init):
                    self.assignment_map[func_name][name] = self.generator.visit(init)


def resolve_sketch(sketch_code: str, synthesis_vars: list[str]) -> SynthesisFacts:
    c_code = run_sketch_solver_as_c(sketch_code)
    if c_code is None:
        return SynthesisFacts({})

    sketch_template_indices = [
        c_code.find(f"void {var}_sketch") for var in synthesis_vars
    ]

    start_idx = min(sketch_template_indices)
    end_idx = max(sketch_template_indices)
    end_idx = c_code.find("}", end_idx) + 1

    result_code = c_code[start_idx:end_idx].replace("int&", "int")
    print(result_code, end="\n\n\n")

    # ast = parse_file(filename, use_cpp=True)
    ast = CParser().parse(result_code, "tmp.c")
    visitor = FuncDefVisitor()
    visitor.visit(ast)

    return SynthesisFacts(
        {k.removesuffix("_sketch"): v for k, v in visitor.result_map.items()}
    )
