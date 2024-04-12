import ast
from itertools import chain, groupby
from typing import Optional

from b2s.const import MAGIC_ID, VAR_NAME
from b2s.converters.naive_sketch_generator import NaiveSketchGenerator
from b2s.input_config import InputConfig
from b2s.program_template import ProgramTemplate
from b2s.solvers.sketch import *
from b2s.utils import *


def fill_in_unknowns(
    func: ast.FunctionDef, syn_result: SynthesisFacts
) -> ast.FunctionDef:
    class CompleteSketchVisitor(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.Assign:
            match node:
                case ast.Assign(
                    value=ast.Call(func=ast.Name(id="synthesize")),
                    targets=[ast.Name(id=var_name)],
                ):
                    if var_name in syn_result:
                        expr = parse_stmt(syn_result[var_name])
                        assert isinstance(
                            expr, ast.Expr
                        ), "Expected an expression as synthesis fact"
                        return ast.Assign(
                            targets=node.targets,
                            value=expr.value,
                        )
            return node

    return ast.fix_missing_locations(CompleteSketchVisitor().visit(func))


class NaiveNonnestedConverter:
    """Converts a batch-processing program with single non-nested loop to
    a stream-processing program.
    """

    def convert(self, src: str) -> str:
        module = ast.parse(src)
        docstr = get_docstring(module)
        assert docstr is not None, "Cannot find docstring"
        spec = InputConfig.load_from_docstring(docstr)
        func: ast.FunctionDef = next(  # type: ignore
            filter(  # type: ignore
                lambda node: isinstance(node, ast.FunctionDef)
                and node.name == spec.func,
                module.body,
            )
        )

        assert not get_variables_in_scope(func).intersection(
            VAR_NAME.RESERVERD_NAMES
        ), "Reserved variable name"

        (
            init_code,
            [loop_code],
            post_code,
        ) = NaiveNonnestedConverter._split_program_naive(func)

        annotated_nodes = self.annotate_source_program(
            spec, init_code, loop_code, post_code
        )
        stream_func = self.convert_annotated_to_stream_program(
            spec, func, annotated_nodes
        )

        sketch = NaiveSketchGenerator().generate(spec, annotated_nodes)
        synthesis_result = resolve_sketch(sketch, annotated_nodes.synthesis_vars)

        return ast.unparse(fill_in_unknowns(stream_func, synthesis_result))

    def annotate_source_program(
        self,
        spec: ProgramSpec,
        init_code: list[ast.stmt],
        loop_code: ast.For,
        post_code: list[ast.stmt],
    ) -> ProgramTemplate[ast.stmt]:
        def _rewrite_phase_1_map(name: str) -> Optional[str]:
            """rewrite variable names to `prev_*` and create `prev_out`"""
            match name:
                case MAGIC_ID.RETURN:
                    return VAR_NAME.PREVIOUS_OUTPUT
                case VAR_NAME.INPUT_STREAM:
                    return VAR_NAME.PREVIOUS_LIST
                case _:
                    return None

        annotated_program = ProgramTemplate[ast.stmt].create()

        annotated_program.list_split.append(
            parse_stmt(
                f"*{VAR_NAME.PREVIOUS_LIST}, {VAR_NAME.CURRENT_ELEMENT} = {VAR_NAME.INPUT_STREAM}"
            )
        )

        annotated_program.prev_init += [
            rewrite_variable_name(_rewrite_phase_1_map, node) for node in init_code
        ]
        annotated_program.prev_loop += [
            rewrite_variable_name(_rewrite_phase_1_map, loop_code)
        ]
        annotated_program.prev_post += [
            rewrite_variable_name(_rewrite_phase_1_map, node) for node in post_code
        ]

        # assign state (relational spec) variables
        annotated_program.prev_rel_init += [
            parse_stmt(f"{name} = {expr}") for name, expr in spec.relational_sig.items()
        ]

        # recover variables updated in the loop (inverse operation)
        loop_left_vars = extract_left_values(loop_code)
        loop_decl_vars = get_variables_in_scope(loop_code.target)
        for left_var in loop_left_vars:
            if left_var in loop_decl_vars:
                continue  # exclude loop (declared) variable
            if all(
                check_if_new_decl(s) is not False
                for s in filter_statements_by_vars(loop_code.body, {left_var})
            ):
                continue  # exclude new variable (usually some index-based temp variable)

            var_name = f"{VAR_NAME.CURRENT_PREFIX}{left_var}"
            annotated_program.recover_loop_state.append(
                parse_stmt(
                    f"{var_name} = synthesize({{{', '.join(spec.params)}}}, {var_name} == {left_var})"
                )
            )
            annotated_program.synthesis_vars.append(var_name)

        # update state variables (inductive relation)
        for var, term in spec.relational_sig.items():
            # skip if term is not in the form of `f(..., _prev_xs, ...)`
            term_expr = parse_stmt(term)
            call_found = False
            for node in ast.walk(term_expr):
                match node:
                    case ast.Call(func=ast.Name(), args=args):
                        for name in chain.from_iterable(ast.walk(arg) for arg in args):
                            if (
                                isinstance(name, ast.Name)
                                and name.id == VAR_NAME.PREVIOUS_LIST
                            ):
                                call_found = True
                if call_found:
                    break
            else:
                continue

            new_term_expr = rewrite_variable_name(
                lambda name: VAR_NAME.INPUT_STREAM
                if name == VAR_NAME.PREVIOUS_LIST
                else None,
                term_expr,
            )
            var_name = swap_prefix(
                var, VAR_NAME.PREVIOUS_PREFIX, VAR_NAME.CURRENT_PREFIX
            )
            annotated_program.update_inductive_var.append(
                parse_stmt(
                    f"{var_name} = synthesize({{{var}}}, {var_name} == {ast.unparse(new_term_expr)})"
                )
            )
            annotated_program.synthesis_vars.append(var_name)

        # unroll loop body
        for_decl_var, *rest_for_decl_vars = list(
            extract_top_var_decls(loop_code.target)
        )
        if not all(v in spec.loop_var_map for v in rest_for_decl_vars):
            assert (
                False
            ), "only single iteration variable is supported; otherwise, add them to loop_var_map"
        if (
            for_decl_var not in spec.loop_var_map
            and f"{VAR_NAME.INPUT_STREAM}[{for_decl_var}]" not in spec.loop_var_map
        ):
            if check_is_foreach_loop(loop_code):
                spec.loop_var_map[for_decl_var] = VAR_NAME.CURRENT_ELEMENT
            elif check_is_indexed_loop(loop_code):
                spec.loop_var_map[
                    f"{VAR_NAME.INPUT_STREAM}[{for_decl_var}]"
                ] = VAR_NAME.CURRENT_ELEMENT
            else:
                assert False, "Cannot infer loop variable mapping"
            print(f"Warning: inferred {spec.loop_var_map}")

        def _rewrite_phase_2_subscript(subscript: ast.Subscript) -> ast.expr:
            match subscript:
                case ast.Subscript(slice=ast.Tuple(elts=elts)):
                    for i, elt in enumerate(elts):
                        if isinstance(elt, ast.Subscript):
                            elts[i] = _rewrite_phase_2_subscript(elt)
                    return ast.fix_missing_locations(subscript)
                case _:
                    src_subscript = ast.unparse(subscript)
                    if src_subscript in spec.loop_var_map:
                        return parse_term(spec.loop_var_map[src_subscript])
            return None

        def _rewrite_phase_2_map(name: str) -> Optional[str]:
            match name:
                case _ if name in spec.loop_var_map:
                    return spec.loop_var_map[name]
                case _ if name in loop_left_vars:
                    return f"{VAR_NAME.CURRENT_PREFIX}{name}"

            return None

        annotated_program.unrolled_loop += [
            rewrite_variable_name(
                _rewrite_phase_2_map,
                rewrite_subscript(_rewrite_phase_2_subscript, node),
            )
            for node in loop_code.body
        ]

        # reapply post-processing code (by swapping `prev_*` to `cur_*`)
        def _rewrite_call_phase_3_map(node: ast.Call) -> Optional[ast.expr]:
            call_term_str = ast.unparse(node)

            for var in spec.relational_sig:
                if call_term_str != spec.relational_sig[var]:
                    continue
                return ast.fix_missing_locations(
                    ast.Name(
                        f"{swap_prefix(var, VAR_NAME.PREVIOUS_PREFIX, VAR_NAME.CURRENT_PREFIX)}",
                        ast.Load(),
                    )
                )

            return None

        def _rewrite_phase_3_map(name: str) -> Optional[str]:
            match name:
                case _ if name in loop_left_vars:
                    return f"{VAR_NAME.CURRENT_PREFIX}{name}"
                case VAR_NAME.PREVIOUS_OUTPUT:
                    return VAR_NAME.CURRENT_OUTPUT
            return None

        annotated_program.post += [
            rewrite_variable_name(
                _rewrite_phase_3_map,
                rewrite_call(
                    rewrite_variable_name(_rewrite_phase_1_map, node),
                    _rewrite_call_phase_3_map,
                ),
            )
            for node in post_code
        ]
        annotated_program.post += [parse_stmt(f"return {VAR_NAME.CURRENT_OUTPUT}")]

        return annotated_program

    def convert_annotated_to_stream_program(
        self,
        spec: ProgramSpec,
        func: ast.FunctionDef,
        annotated_program: ProgramTemplate[ast.stmt],
    ) -> ast.FunctionDef:
        stream_func_returns = [
            swap_prefix(v, VAR_NAME.PREVIOUS_PREFIX, VAR_NAME.CURRENT_PREFIX)
            for v in spec.relational_sig
        ] + [VAR_NAME.CURRENT_OUTPUT]

        stream_func_body = annotated_program.to_stream_version_list()[:-1]
        stream_func_body.append(parse_stmt("return " + ", ".join(stream_func_returns)))
        stream_func_args = (
            ast.parse(f"def _({', '.join(spec.params)}): pass").body[0].args  # type: ignore
        )
        stream_func = ast.fix_missing_locations(
            ast.FunctionDef(
                f"{spec.func_name}_stream",
                body=stream_func_body,
                args=stream_func_args,
                returns=func.returns,
                decorator_list=func.decorator_list,
            )
        )

        return stream_func

    @staticmethod
    def _split_program_naive(
        func: ast.FunctionDef,
    ) -> tuple[list[ast.stmt], list[ast.For], list[ast.stmt]]:
        """Splits the program into three parts: initialization, aggregation, and post-processing."""
        for_node = next(filter(lambda x: isinstance(x, ast.For), func.body), None)
        if for_node is None:
            raise ValueError("No loop found in function body.")

        chunks = tuple(
            list(group) for _, group in groupby(func.body, lambda x: x != for_node)
        )
        assert len(chunks) == 3, "Program must have exactly three parts."
        return chunks  # type: ignore
