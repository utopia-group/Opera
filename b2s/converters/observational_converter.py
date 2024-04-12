import copy
import logging
import time
from _ast import Call, For, FunctionDef
from dataclasses import dataclass, replace
from typing import Any, Generic, Optional, Self, Sequence, TypeVar

import b2s.ast_extra as ast
from b2s.const import SYN_OUTPUT_VAR_KEY, VAR_NAME
from b2s.converters.irs.inference import infer_imperative_relational_signature
from b2s.expr_template import ExprTemplate
from b2s.input_config import ImperativeRelationalSignature, InputConfig
from b2s.sketch_types import ASTSketch, ASTUnknownSpec, EnvState, ImperativeUnknownSpec, Sketch, UnknownSpec, Value
from b2s.synthesizers.enum_syn import synthesize_from_io
from b2s.synthesizers.qe_syn import ImperativeQEGrammarConstructor
from b2s.synthesizers.template_enum_syn import synthesize_from_templates
from b2s.utils import (
    check_if_new_decl,
    extract_left_values,
    filter_statements_by_vars,
    find_function_by_name,
    get_docstring,
    get_variables_in_scope,
    parse_stmt,
    parse_term,
    rewrite_variable_name,
    subdict,
)

USE_IRS_INFERENCE = False


@dataclass(slots=True, frozen=True)
class ExecResult:
    """
    Observational execution result of a program.

    Attributes:
        output: The output value of the program.
        final_state: The local variable states after the loop.
        unknowns: Maps unknown variable names to their values.
    """

    output: Value
    unknowns: dict[str, Any]
    final_state: EnvState


T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class IncrementalExecResult(Generic[T]):
    fst: ExecResult
    snd: ExecResult
    delta: T


@dataclass(slots=True, frozen=True)
class ExprResult:
    """
    Synthesis result of a single expression.
    """

    expr: str


@dataclass(slots=True, frozen=True)
class StmtsResult:
    """
    Synthesis result of a fold loop that simulates tuple assignment.
    """

    stmts: list[str]


SynResult = ExprResult | StmtsResult


class SketchifyFunction(ast.NodeTransformer):
    mapping: dict[str, ASTUnknownSpec]
    _next_unk_id: int

    def __init__(self) -> None:
        super().__init__()

        self.mapping = {}
        self._next_unk_id = 0

    def _fresh_unk_id(self) -> str:
        self._next_unk_id += 1
        return f"unk_{self._next_unk_id}__"

    def sketchify(
        self, node: ast.FunctionDef, rs: ImperativeRelationalSignature
    ) -> ASTSketch:
        for k, v in rs.items():
            if not isinstance(t := parse_term(v), ast.Call):
                continue

            if any(unk.equivalent_expr == t for unk in self.mapping.values()):
                continue

            unk_id = self._fresh_unk_id()
            self.mapping[unk_id] = UnknownSpec(unk_id, t, None)
        return Sketch(self.visit(copy.deepcopy(node)), self.mapping)

    def visit_For(self, node: For) -> Any:
        if node.orelse:
            raise ValueError("Cannot handle for-else statement")

        match node.iter:
            case ast.Name(id=VAR_NAME.INPUT_STREAM):
                pass
            case ast.Call(
                func=ast.Name(id="range"),
                args=[
                    ast.Call(
                        func=ast.Name(id="len"),
                        args=[ast.Name(id=VAR_NAME.INPUT_STREAM)],
                    )
                ],
            ):
                raise NotImplementedError(
                    f"range(len({VAR_NAME.INPUT_STREAM})) is yet supported"
                )
            case _:
                raise NotImplementedError(f"TBD: {ast.unparse(node.iter)}")

        unk_id = self._fresh_unk_id()
        self.mapping[unk_id] = UnknownSpec(unk_id, node, None)
        return ast.Call(func=ast.Name(id=unk_id), args=[], keywords=[])

    def visit_Call(self, node: Call) -> Any:
        match node:
            case Call(func=ast.Name(), args=args) if any(
                isinstance(arg, ast.Name) and arg.id == VAR_NAME.INPUT_STREAM
                for arg in args
            ):
                unk_id = self._fresh_unk_id()
                self.mapping[unk_id] = UnknownSpec(unk_id, node, None)
                return ast.Call(func=ast.Name(id=unk_id), args=[], keywords=[])

        # functional call that does not use the input stream
        new_args = [self.visit(arg) for arg in node.args]
        return ast.Call(func=node.func, args=new_args, keywords=[])


class CaptureFunctionState(ast.NodeTransformer):
    IRRELEVANT_KEYS = {
        VAR_NAME.INPUT_STREAM,
        VAR_NAME.STATE_AFTER_LOOP_PREUNROLL,
        VAR_NAME.STATE_AFTER_LOOP_POSTUNROLL,
        VAR_NAME.STATE_BEFORE_LOOP,
        VAR_NAME.PRE_STATES,
        VAR_NAME.POST_PREUNROLL_STATES,
        VAR_NAME.POST_POSTUNROLL_STATES,
        VAR_NAME.STATE_UNKNOWNS,
    }

    @staticmethod
    def _emit_pop_stmt(var: str, keys: set[str]) -> list[ast.stmt]:
        return [parse_stmt(f"{var}.pop({repr(k)}, None)") for k in keys]

    def __init__(
        self,
        sketch: Sketch,
        rel_sig: ImperativeRelationalSignature,
    ) -> None:
        self.func_name = sketch.sketch.name
        self.sketch = copy.deepcopy(sketch.sketch)
        self.mapping = sketch.unknowns
        self.rel_sig = rel_sig

    def run(self) -> ast.FunctionDef:
        return self.visit(self.sketch)

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        if node.name != self.func_name:
            return node
        node.body.insert(0, parse_stmt(f"{VAR_NAME.STATE_UNKNOWNS} = dict()"))
        return super().generic_visit(ast.fix_missing_locations(node))

    def visit_Return(self, node: ast.Return) -> Any:
        if node.value is None:
            return super().visit_Return(node)

        ret_node = self.visit(node.value)
        ret_val_stmt = parse_stmt(f"{VAR_NAME.RET_TEMP_VAR} = {ast.unparse(ret_node)}")
        final_str = (
            "{"
            + ", ".join(
                [
                    f"{repr(arg_name)}:{expr if expr != 'return' else VAR_NAME.RET_TEMP_VAR}"
                    for arg_name, expr in self.rel_sig.items()
                ]
            )
            + "}"
        )
        return [
            ast.fix_missing_locations(ret_val_stmt),
            ast.fix_missing_locations(
                parse_stmt(
                    f"return {VAR_NAME.RET_TEMP_VAR}, {VAR_NAME.STATE_UNKNOWNS}, {final_str}"
                )
            ),
        ]

    def visit_Call(self, node: Call) -> Any:
        match node:
            case ast.Call(func=ast.Name(id=name)) if name in self.mapping:
                dest_node = self.mapping[name].equivalent_expr

                match dest_node:
                    case ast.For():
                        unk_key = f"{VAR_NAME.STATE_UNKNOWNS}[{repr(name)}]"
                        return [
                            dest_node,
                            ast.fix_missing_locations(
                                parse_stmt(f"{unk_key} = deepcopy(dict(vars()));"),
                            ),
                        ] + self._emit_pop_stmt(
                            unk_key,
                            self.IRRELEVANT_KEYS,
                        )
                    case ast.Call():
                        expr = ast.unparse(dest_node)
                        return ast.fix_missing_locations(
                            parse_term(
                                f"_store_state({VAR_NAME.STATE_UNKNOWNS}, {repr(name)}, {expr})"
                            )
                        )

        return super().generic_visit(node)


class FillInSketch(ast.NodeTransformer):
    def __init__(
        self,
        sketch: Sketch,
        mapping: dict[str, SynResult],
        rel_sig: ImperativeRelationalSignature,
    ) -> None:
        super().__init__()
        self.sketch_func = sketch.sketch
        self.sketch_unks = sketch.unknowns
        self.mapping = mapping
        self.rel_sig = rel_sig

    def run(self) -> ast.FunctionDef:
        return self.visit(self.sketch_func)

    def visit_FunctionDef(self, node: FunctionDef) -> Any:
        func = super().generic_visit(node)
        assert isinstance(func, ast.FunctionDef)

        new_args = list(self.rel_sig.keys()) + [VAR_NAME.CURRENT_ELEMENT]
        func.args.args = [
            ast.arg(arg=arg_name, annotation=None) for arg_name in new_args
        ]
        func.args.kwonlyargs = []
        func.args.kw_defaults = []
        func.args.defaults = []
        return func

    def visit_Call(self, node: Call) -> Any:
        match node:
            case Call(func=ast.Name(id=name)) if name in self.mapping:
                match self.mapping[name]:
                    case ExprResult(expr=expr):
                        return ast.fix_missing_locations(parse_term(expr))
                    case StmtsResult(stmts=stmts):
                        return [
                            ast.fix_missing_locations(parse_stmt(stmt))
                            for stmt in stmts
                        ]
        return super().generic_visit(node)

    def visit_Return(self, node: ast.Return) -> Any:
        new_ret = super().generic_visit(node)
        assert isinstance(new_ret, ast.Return)
        ret_items = []

        for arg_name, expr in self.rel_sig.items():
            match parse_term(expr):
                case ast.Name(id=VAR_NAME.RET_TEMP_VAR):
                    ret_items.append(
                        ast.unparse(new_ret.value) if new_ret.value else "None"
                    )
                case ast.Name(id=arg_name):
                    ret_items.append(arg_name)
                case ast.Call():
                    for unk_id, unk in self.sketch_unks.items():
                        if ast.unparse(unk.equivalent_expr) == expr:
                            soln = self.mapping.get(unk_id)
                            assert isinstance(soln, ExprResult)
                            ret_items.append(soln.expr)
                            break
                    else:
                        raise NotImplementedError

                case _:
                    raise NotImplementedError

        return ast.fix_missing_locations(parse_stmt(f"return {', '.join(ret_items)}"))


class ToOnline(ast.NodeTransformer):
    def __init__(
        self,
        target_func: str,
        loop_idx_to_vars: dict[int, list[str]],
        aux_exprs: dict[str, str],
        var_exprs: dict[str, str],
        rel_spec: ImperativeRelationalSignature,
    ):
        self.target_func = target_func
        self.loop_idx_to_vars = loop_idx_to_vars
        self.var_exprs = var_exprs
        self.loop_idx = 0
        self.reversed_aux_exprs = {v: k for k, v in aux_exprs.items()}
        self.rel_spec = rel_spec

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        if node.name != self.target_func:
            return node

        params = [*self.rel_spec.keys(), VAR_NAME.CURRENT_ELEMENT]
        node.args.args = [ast.arg(arg=p) for p in params]
        for aux_var in self.reversed_aux_exprs.values():
            node.body.insert(0, parse_stmt(f"{aux_var} = {self.var_exprs[aux_var]}"))
        return super().generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
        match node.iter:
            case ast.Name(id=VAR_NAME.INPUT_STREAM):
                pass
            case ast.Call(
                func=ast.Name(id="range"),
                args=[
                    ast.Call(
                        func=ast.Name(id="len"),
                        args=[ast.Name(id=VAR_NAME.INPUT_STREAM)],
                    )
                ],
            ):
                raise NotImplementedError("range(len(INPUT_LIST)) is not supported rn!")
            case _:
                return super().generic_visit(node)

        stmts = [
            parse_stmt(f"{ex} = {self.var_exprs[ex]}")
            for ex in self.loop_idx_to_vars[self.loop_idx]
        ]
        unrolled_body = ast.fix_missing_locations(  # type: ignore
            rewrite_variable_name(
                lambda x: x if x != node.target.id else VAR_NAME.CURRENT_ELEMENT, node  # type: ignore
            )
        ).body

        self.loop_idx += 1
        return stmts + unrolled_body

    def visit_Call(self, node: ast.Call) -> Any:
        call_expr_str = ast.unparse(node)
        if call_expr_str in self.reversed_aux_exprs:
            return parse_term(self.reversed_aux_exprs[call_expr_str])

        return super().generic_visit(node)

    def visit_Return(self, node: ast.Return) -> Any:
        assert node.value is not None, "Return value must be non-None"
        params = [ast.unparse(node.value), *self.rel_spec.values()]
        params = list(filter(lambda p: p != VAR_NAME.RET_TEMP_VAR, params))
        return super().generic_visit(parse_stmt(f"return {', '.join(params)}"))


def _store_state(st: dict[str, Any], k: str, v: Any) -> Any:
    st[k] = v
    return v


class ObservationalConverter:
    """
    Converts an offline program to an online program with observational equivalence.
    """

    code_cache: dict[str, Any]

    def __init__(self, num_workers: int, **_) -> None:
        self.num_workers = num_workers
        self.code_cache = {}

    def collect_program_state(
        self,
        func: ast.FunctionDef,
        base_module: ast.Module,
        params: Sequence[tuple[Value, ...]],
    ):
        """
        Collects input-output examples from the given annotated function.

        The given function should return a tuple of the form
            (output, pre_states, post_states, valuation of aux_exprs, valuation of rel_specs).
        """

        old_func = find_function_by_name(base_module, func.name)
        if old_func is not None and old_func is not func:
            base_module.body.remove(old_func)

        base_module.body.append(func)
        src = ast.unparse(base_module)
        if src not in self.code_cache:
            code = compile(base_module, filename="<ast>", mode="exec")
            self.code_cache[src] = code

        local_vars: dict[Any, Any] = {
            "_store_state": _store_state,
        }
        exec("from copy import deepcopy", local_vars)
        exec(self.code_cache[src], local_vars)

        assert func.name in local_vars, "Function not found in the module"
        func_callable = local_vars[func.name]
        return [func_callable(*param) for param in params]

    def collect_exec_results(
        self,
        func: ast.FunctionDef,
        base_module: ast.Module,
        input_list: Sequence[tuple[Value, ...]],
    ) -> list[IncrementalExecResult]:
        results = self.collect_program_state(func, base_module, input_list)

        states = [
            IncrementalExecResult(
                ExecResult(*init_raw),
                ExecResult(*comp_raw),
                tail_value,
            )
            for init_raw, comp_raw, tail_value in zip(
                results[::2],
                results[1::2],
                map(lambda xs: xs[0][-1], input_list[1::2]),
            )
        ]

        return states

    def enum_solve(
        self: Self,
        unk_specs: list[ASTUnknownSpec],
        expr_templates: dict[str, list[ExprTemplate]] | None,
    ) -> dict[str, SynResult]:
        """
        Solves the given synthesis specifications.

        Returns:
            A mapping from the name of unknowns to its synthesized program.
        """

        if expr_templates is None:
            expr_templates = {}

        syn_results: dict[str, SynResult] = {}
        for spec in unk_specs:
            assert spec.io_examples is not None
            start_time = time.time()

            template_key = (
                (spec.variable, spec.variable)
                if isinstance(spec, ImperativeUnknownSpec)
                else (spec.unk_id, ast.unparse(spec.equivalent_expr))
            )
            use_template = template_key[0] in expr_templates

            syn_result = (
                synthesize_from_templates(
                    spec.io_examples,
                    expr_templates[template_key[0]],
                    template_key[0],
                    self.num_workers,
                )
                if use_template
                else synthesize_from_io(spec.io_examples)
            )

            assert isinstance(syn_result, str) or syn_result is None

            if syn_result is None:
                logging.error(
                    f"Failed to synthesize {template_key[1]} from {'template' if use_template else 'IO'}"
                )
            else:
                logging.info(
                    f"Synthesized {template_key[1]} in {time.time() - start_time :.2f}s "
                    f"from {'template' if use_template else 'IO'}"
                )

                match spec:
                    case ImperativeUnknownSpec(variable=v):
                        assert isinstance(
                            syn_results.setdefault(spec.unk_id, StmtsResult([])),
                            StmtsResult,
                        )
                        syn_results[spec.unk_id].stmts.append(f"{v} = {syn_result}")  # type: ignore

                    case UnknownSpec():
                        syn_results[spec.unk_id] = ExprResult(syn_result)

        return syn_results

    def convert(self, src: str) -> Optional[str]:
        """
        Converts an offline program to an online program with observational equivalence.
        """
        module = ast.parse(src)
        docstr = get_docstring(module)

        assert docstr is not None, "Cannot find docstring"
        input_truth = InputConfig.load_from_docstring(docstr)
        func = find_function_by_name(module, input_truth.func)
        assert func is not None, f"Function {input_truth.func} not found"

        rs = (
            infer_imperative_relational_signature(func)
            if USE_IRS_INFERENCE or not input_truth.relational_sig
            else input_truth.relational_sig
        )
        logging.info("*" * 62)
        logging.info("Inferred relational signature:")
        pad_width = max(map(len, rs.keys()))
        for k, v in rs.items():
            logging.info(f"\t{k.ljust(pad_width)} -> {v}")
        logging.info("*" * 62)

        input_list = [
            ([1.0, 2.0, 3.0, 5.0],),
            ([1.0, 2.0, 3.0, 5.0, 4.0],),
            ((1, 2, 3, 4, 5, 6, 7, 8, 9),),
            ((1, 2, 3, 4, 5, 6, 7, 8, 9, 10),),
            ((11, 12, 13, 14),),
            ((11, 12, 13, 14, 15),),
            ((1, 4),),
            ((1, 4, 7),),
            ((11, 12, 23),),
            ((11, 12, 23, 9),),
            ((1, 5, 3, 4, 2, 6),),
            ((1, 5, 3, 4, 2, 6, 7),),
        ]

        sketch = SketchifyFunction().sketchify(func, rs)

        annotated_func = CaptureFunctionState(sketch, rs).run()
        exec_results = self.collect_exec_results(annotated_func, module, input_list)

        unk_specs: list[ASTUnknownSpec] = []

        for unk_id, spec in sketch.unknowns.items():
            match spec.equivalent_expr:
                case ast.Call():
                    key = (
                        unk_id
                        if unk_id in exec_results[0].snd.unknowns
                        else next(
                            filter(
                                lambda p: p[1] == ast.unparse(spec.equivalent_expr),
                                rs.items(),
                            )
                        )[0]
                    )

                    unk_specs.append(
                        replace(
                            spec,
                            io_examples=[
                                (
                                    EnvState(
                                        subdict(
                                            {
                                                **r.fst.final_state,
                                                VAR_NAME.CURRENT_ELEMENT: r.delta,
                                            },
                                            list(set(rs.keys()) | {VAR_NAME.CURRENT_ELEMENT}),
                                        )
                                    ),
                                    r.snd.unknowns[key]
                                    if key
                                    in r.snd.unknowns  # exprs extracted from the input
                                    else r.snd.final_state[
                                        key
                                    ],  # additional exprs from rel sig
                                )
                                for r in exec_results
                            ],
                        )
                    )

                case ast.For() as loop:
                    loop_left_vars = extract_left_values(loop)
                    loop_decl_vars = get_variables_in_scope(loop.target)

                    for left_var in loop_left_vars:
                        # exclude loop variables
                        if left_var in loop_decl_vars:
                            continue

                        # exclude new variables (usually some index-based temp variable)
                        # that are out of scope after the loop
                        if all(
                            check_if_new_decl(s) is not False
                            for s in filter_statements_by_vars(loop.body, {left_var})
                        ):
                            logging.debug(f"Skipping {left_var} as it is out of scope")
                            pass

                        allowed_variables = (
                            (
                                set(rs.keys())
                                | {VAR_NAME.CURRENT_ELEMENT}
                                | exec_results[0].fst.unknowns[unk_id].keys()
                            )
                            - set(loop_left_vars)
                            - set(loop_decl_vars)
                        )

                        unk_specs.append(
                            ImperativeUnknownSpec(
                                unk_id=spec.unk_id,
                                equivalent_expr=spec.equivalent_expr,
                                variable=left_var,
                                io_examples=[
                                    (
                                        EnvState(
                                            subdict(
                                                {
                                                    **r.snd.unknowns[unk_id],
                                                    **r.fst.final_state,
                                                    VAR_NAME.CURRENT_ELEMENT: r.delta,
                                                },
                                                allowed_variables,
                                            )
                                        ),
                                        r.snd.unknowns[unk_id][left_var],
                                    )
                                    for r in exec_results
                                ],
                            )
                        )

        grammar_ctr = ImperativeQEGrammarConstructor()
        grammar = grammar_ctr.construct_grammar(func, rs, unk_specs)
        syn_res = self.enum_solve(unk_specs, grammar)
        return ast.unparse(FillInSketch(sketch, syn_res, rs).run())
