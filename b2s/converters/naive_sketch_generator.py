import ast

from b2s.const import VAR_NAME
from b2s.input_config import InputConfig
from b2s.program_template import ProgramTemplate
from b2s.utils import *

RESERVED_VARS = {"_prev_xs", "xs", "len"}
TRANSLATE_TRUTHS = {
    "len(_prev_xs)": "(N-1)",
    "len(xs)": "(N)",
    "_prev_xs[-1]": "xs[N-2]",
    "_prev_xs[-2]": "xs[N-3]",
}


def _replace_map(s: str, m: dict[str, str]) -> str:
    for k, v in m.items():
        s = s.replace(k, v)
    return s


INDENTED_NEWLINE = "\n" + " " * 4


SIMPLE_EXPR_GENERATOR = """\
generator int simple_expr(fun choices, int bnd) {
    assert bnd > 0;
    if (??) {
        return choices();
    } else {
        return {| choices() (+ | - | * | /) simple_expr(choices, bnd-1) |};
    }
}"""

SYNTHESIS_MAIN_TEMPLATE = """\
pragma options "--bnd-arr-size 5";
harness void main(int N, int[N] xs, {args}) {{
    assume N > 3;
    int _x = xs[N-1];
    {assumptions}
    {recover_prev_states}
    {c_stmts}
    {synthesis_decls}
    {assertions}
}}"""

SYNTHESIS_PROBLEM_TEMPLATE = """\
{type} {name}_sketch({args}) {{
    generator {type} {name}_gen() {{
        return {{| {regex} |}};
    }}
    return simple_expr({name}_gen, {depth});
}}"""

SYNTHESIS_VAR_DECL_TEMPLATE = "{type} {name} = {name}_sketch({args});"


class NaiveSketchGenerator:
    """Generate sketch template for synthesis problems in an annotated program."""

    def generate(
        self, spec: InputConfig, annotated_program: ProgramTemplate[ast.stmt]
    ) -> str:
        # all variables in scope for the main harness function
        vars_in_scope = (
            set(
                itertools.chain.from_iterable(
                    get_variables_in_scope(stmt)
                    for stmt in (
                        annotated_program.prev_post + annotated_program.prev_rel_init
                    )
                )
            )
        ) - RESERVED_VARS

        main_args = [f"int {v}" for v in vars_in_scope]
        assumptions = [
            _replace_map(s, TRANSLATE_TRUTHS)
            for s in NaiveSketchGenerator._build_assumptions(
                annotated_program.prev_post + annotated_program.prev_rel_init
            )
        ]
        recover_prev_states = [
            _replace_map(f"{ast.unparse(s)};", TRANSLATE_TRUTHS)
            for s in annotated_program.prev_rel_init
        ]

        synthesis_variable_declarations = []
        synthesis_variable_templates = []
        assertions = []

        # format synthesis problems
        for stmt in (
            annotated_program.recover_loop_state
            + annotated_program.update_inductive_var
        ):
            (
                syn_name,
                var_names,
                constraint,
            ) = NaiveSketchGenerator._extract_synthesis_problem(stmt)

            main_decl, syn_decl = NaiveSketchGenerator._format_synthesis_problem(
                syn_name, var_names, "int", 3
            )

            synthesis_variable_templates.append(syn_decl)
            synthesis_variable_declarations.append(main_decl)
            assertions.append(
                f"assert {_replace_map(ast.unparse(constraint), TRANSLATE_TRUTHS)};"
            )

        # build main harness function
        post_code_str = list(
            map(
                ast.unparse,
                annotated_program.prev_post,
            )
        )
        post_code_str = [f"{_replace_map(s, TRANSLATE_TRUTHS)};" for s in post_code_str]

        main_code = SYNTHESIS_MAIN_TEMPLATE.format(
            args=", ".join(main_args),
            assumptions=INDENTED_NEWLINE.join(assumptions),
            recover_prev_states=INDENTED_NEWLINE.join(recover_prev_states),
            synthesis_decls=INDENTED_NEWLINE.join(synthesis_variable_declarations),
            c_stmts=INDENTED_NEWLINE.join(post_code_str),
            assertions=INDENTED_NEWLINE.join(assertions),
        )

        return "\n".join(
            [
                SIMPLE_EXPR_GENERATOR,
                *synthesis_variable_templates,
                main_code,
            ]
        )

    @staticmethod
    def _extract_synthesis_problem(
        syn_stmt: ast.stmt,
    ) -> tuple[str, list[str], ast.expr]:
        match syn_stmt:
            case ast.Assign(
                value=ast.Call(
                    func=ast.Name(id="synthesize"),
                    args=[ast.Set(elts=var_name_exprs), constraint],
                ),
                targets=[ast.Name(id=var_name)],
            ):
                assert all(isinstance(v, ast.Name) for v in var_name_exprs)
                var_names = [
                    var.id for var in var_name_exprs if isinstance(var, ast.Name)
                ]
                return var_name, var_names, constraint

        raise ValueError("Invalid synthesis statement", ast.unparse(syn_stmt))

    @staticmethod
    def _format_synthesis_problem(
        syn_name: str,
        var_names: list[str],
        syn_type: str,
        depth: int,
    ) -> tuple[str, str]:
        """Format a synthesis problem into a sketch template.
        Warning: this won't process constraint!
        Returns:
            - a function that returns the sketch template for the synthesis problem
            - a string that declares the synthesis problem in the main harness function
        """
        main_decl = SYNTHESIS_VAR_DECL_TEMPLATE.format(
            type=syn_type, name=syn_name, args=", ".join(var_names)
        )

        prob_decl = SYNTHESIS_PROBLEM_TEMPLATE.format(
            type=syn_type,
            name=syn_name,
            args=", ".join([f"int {v}" for v in var_names]),
            depth=depth,
            regex=" | ".join(var_names + ["??"]),
        )

        return main_decl, prob_decl

    @staticmethod
    def _build_assumptions(post_code: list[ast.stmt]) -> list[str]:
        division_terms: set[tuple[str, str]] = set(
            map(
                lambda ls: tuple(map(ast.unparse, ls)),  # type: ignore
                itertools.chain.from_iterable(
                    find_division_terms(c) for c in post_code
                ),
            )
        )
        divisors = set(map(lambda t: t[1], division_terms))

        # divide by zero assumptions
        assumptions = [f"assume {d} != 0;" for d in divisors]

        # mimic floating point division by setting mod to 0
        assumptions += [f"assume (({x}) % ({d})) == 0;" for x, d in division_terms]
        return assumptions
