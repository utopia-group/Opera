# ruff: noqa: F403, F405
from __future__ import annotations

import ast
import copy
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import singledispatch
from itertools import product
from typing import Any, Generic, Literal, Self, Sequence, TypeAlias, TypeVar

import sympy as sp
import z3
from frozendict import frozendict
from sympy import parse_expr
from tqdm import tqdm

import b2s.lang as ir
from b2s.const import VAR_NAME
from b2s.expr_template import ConcreteExprGrammar, ExprTemplate
from b2s.lang import Env, Expr, to_value_list
from b2s.lang_symb_exec import (
    VSymbol,
    VSymbolExpr,
    VSymPreludeFunc,
    sym_exec,
    val_to_expr,
)
from b2s.rfs import (
    ImperativeRelationalSignature,
    RelationalSignature,
    RelationalSignatureExpr,
)
from b2s.sketch_types import (
    ASTUnknownSpec,
    ImperativeUnknownSpec,
    IRUnknownSpec,
    UnknownSpec,
)
from b2s.solvers.reduce import run_rlqe, solve_batch_eqns
from b2s.synthesizers.qe_types import *
from b2s.utils import (
    extract_left_values,
    foldr,
    map_expr_to_partial_stream,
    parse_term,
    rewrite_variable_name,
)

OUT_KEYWORD = "___out___"


def to_z3_formula(f: Formula | Expression) -> Any:
    match f:
        case ValueTop():
            return z3.BoolVal(True)
        case ValueBottom():
            return z3.BoolVal(False)
        case Variable(name=name):
            return z3.Real(name)
        case Implication(left=left, right=right):
            return z3.Implies(to_z3_formula(left), to_z3_formula(right))
        case Conjunction(exprs=exprs):
            return z3.And(*map(to_z3_formula, exprs))
        case Disjunction(exprs=exprs):
            return z3.Or(*map(to_z3_formula, exprs))
        case Negation(expr=expr):
            return z3.Not(to_z3_formula(expr))
        case PredicateApp(pred=p_or_f, args=args) | FunctionApp(
            func=p_or_f, args=args
        ) if not isinstance(p_or_f, Variable):
            return to_python_operator(p_or_f)(*map(to_z3_formula, args))
        case Constant(value=value):
            return z3.RealVal(value)

    raise NotImplementedError


@singledispatch
def to_expression(obj: Any) -> Expression:
    raise NotImplementedError


@singledispatch
def to_formula(obj: Any) -> Formula:
    raise NotImplementedError


@singledispatch
def to_python_operator(obj: Any) -> Any:
    raise NotImplementedError


def get_arity(f: FunctionKinds | PredicateKinds) -> int:
    from inspect import signature

    op = to_python_operator(f)
    return len(signature(op).parameters)


@to_python_operator.register
def _function_kind_to_python_operator(func: FunctionKinds) -> Any:
    return FUNCTION_KINDS_TO_OPERATOR[func]


@to_python_operator.register
def _predicate_kind_to_python_operator(pred: PredicateKinds) -> Any:
    return PREDICATE_KINDS_TO_OPERATOR[pred]


@to_expression.register
def _ast_to_expr(expr: ast.expr) -> Expression:
    match expr:
        case ast.Name(id=id):
            return Variable(name=id)

        case ast.BinOp(left=left, op=op_obj, right=right):
            op = AST_TO_FUNCTION_KINDS.get(type(op_obj), None)
            assert op is not None, f"Unsupported op: {ast.unparse(expr)}"
            assert get_arity(op) == 2, f"Invalid arity: {ast.unparse(expr)}"
            return FunctionApp(
                func=op,
                args=(to_expression(left), to_expression(right)),
            )

        case ast.Call(func=func, args=args):
            return FunctionApp(
                func=Variable(ast.unparse(func)),
                args=tuple(to_expression(a) for a in args),
            )

        case ast.Constant(value=value):
            return Constant(value=value)

        case ast.UnaryOp(op=op_obj, operand=operand):
            op = AST_TO_FUNCTION_KINDS.get(type(op_obj), None)
            assert op is not None, f"Unsupported op: {ast.unparse(expr)}"
            assert get_arity(op) == 1, f"Invalid arity: {ast.unparse(expr)}"
            return FunctionApp(
                func=op,
                args=(to_expression(operand),),
            )

    raise ValueError(f"Invalid expr: {expr}")


@to_formula.register
def _ast_to_formula(expr: ast.expr) -> Formula:
    match expr:
        case ast.Compare(left=left, ops=ops, comparators=comparators):
            assert (
                len(ops) == len(comparators) == 1
            ), f"Unsupported compare: {ast.unparse(expr)}"

            op = AST_TO_PREDICATE_KINDS.get(type(ops[0]), None)
            assert op is not None, f"Unsupported op: {ast.unparse(ops[0])}"
            assert get_arity(op) == 2, f"Invalid arity: {ast.unparse(expr)}"
            return PredicateApp(
                pred=op,
                args=(to_expression(left), to_expression(comparators[0])),
            )

    raise ValueError(f"Unsupported expr: {ast.unparse(expr)}")


def _expression_supported_by_reduce(ex: Expression) -> bool:
    match ex:
        case Variable(name=name):
            return True
        case FunctionApp(FunctionKinds.POW, args=(l, r)):
            if str(r) == "(1 / 2)" or str(r) == "0.5":
                return False
            return _expression_supported_by_reduce(
                l
            ) and _expression_supported_by_reduce(r)
        case FunctionApp(func=func, args=args):
            return all(_expression_supported_by_reduce(a) for a in args)
        case Constant(value=value):
            return True

    raise ValueError(f"Invalid expr: {ex}")


class LoopUnrollTransformer(ast.NodeTransformer):
    def __init__(self, depth: int, last_var_name: str | None = None) -> None:
        super().__init__()
        self.depth = depth
        self.last_var_name = last_var_name

    def visit_Call(self, node: ast.Call) -> Any:
        match node:
            case ast.Call(
                func=ast.Name(id="len"), args=[ast.Name(id=VAR_NAME.INPUT_STREAM)]
            ):
                return ast.Constant(value=self.depth)
        return self.generic_visit(node)

    def visit_For(self, node: ast.For) -> Any:
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

        match node.target:
            case ast.Name(id=target_id):
                pass

            case _:
                raise NotImplementedError(f"TBD: {ast.unparse(node.target)}")

        last_item_name = self.last_var_name or f"xs{self.depth - 1}"

        return [
            ast.fix_missing_locations(s)
            for i in range(self.depth - 1)
            for s in [
                rewrite_variable_name(
                    lambda n: f"xs{i}" if n == target_id else None, original_stmt
                )
                for original_stmt in node.body
            ]
        ] + [
            ast.fix_missing_locations(
                rewrite_variable_name(
                    lambda n: last_item_name if n == target_id else None, original_stmt
                )
            )
            for original_stmt in node.body
        ]

    def visit_Return(self, node: ast.Return) -> Any:
        if not node.value:
            return None
        return ast.fix_missing_locations(
            ast.Assign(
                targets=[ast.Name(OUT_KEYWORD, ast.Store())],
                value=self.visit(node.value),
            )
        )


AST_T = TypeVar("AST_T", bound=ast.AST)


def unroll_loop(node: AST_T, depth: int, last_var_name: str | None = None) -> AST_T:
    return LoopUnrollTransformer(depth, last_var_name).visit(
        copy.deepcopy(node),
    )


@dataclass
class RawTemplate:
    guards: set[Formula] = field(default_factory=set)
    updates: set[Formula] = field(default_factory=set)

    def update(self, other: RawTemplate) -> None:
        self.guards |= other.guards
        self.updates |= other.updates


def _raw_template_to_template(
    raw_template: RawTemplate,
    variable: str,
    max_unroll_depth: int,
    rel_sig: RelationalSignature | ImperativeRelationalSignature,
    aux_params: set[str],
    uninterp_to_expr: dict[str, str] | None = None,
    fresh_var_to_exprs: dict[str, tuple[str, ...]] | None = None,
) -> ConcreteExprGrammar:
    if uninterp_to_expr is None:
        uninterp_to_expr = {}
    if fresh_var_to_exprs is None:
        fresh_var_to_exprs = {}

    uninterp_to_exprs = defaultdict(list)
    for k, v in uninterp_to_expr.items():
        for fv in fresh_var_to_exprs:
            if re.search(rf"{fv}\b", v) is not None:
                # since we only handle log/exp, i.e., unary functions
                # we don't have to worry about multiple substitutions
                for fv_expr in fresh_var_to_exprs[fv]:
                    uninterp_to_exprs[k].append(re.sub(rf"{fv}\b", fv_expr, v))
    if uninterp_to_exprs:
        assert uninterp_to_expr

    use_len = VAR_NAME.PREVIOUS_LENGTH in rel_sig

    relevant_updates = [
        to_reduce_formula(update)
        for update in raw_template.updates
        if contains(update, Variable(variable))
    ]
    context: frozendict[str, int] = (
        frozendict({VAR_NAME.PREVIOUS_LENGTH: max_unroll_depth - 1})
        if use_len
        else frozendict()
    )

    def _is_expr_valid(expr: Expression | Formula) -> bool:
        return extract_variables(expr) <= (
            rel_sig.keys()
            | {
                VAR_NAME.CURRENT_ELEMENT,
                VAR_NAME.CURRENT_ELEMENT_1,
                VAR_NAME.CURRENT_ELEMENT_2,
            }
            | uninterp_to_expr.keys()
            | aux_params
        )

    def _process_expr(expr: Expression | Formula) -> tuple[str, ...]:
        s = to_python_str(expr)
        if not uninterp_to_exprs:
            for k, v in uninterp_to_expr.items():
                s = re.sub(rf"\b{k}\b", v, s)
            return (s,)

        result = set()
        for replace_comb in product(*uninterp_to_exprs.values()):
            s_ = s
            for k, v in zip(uninterp_to_exprs.keys(), replace_comb):
                s_ = re.sub(rf"\b{k}\b", v, s_)
            for k, v in uninterp_to_expr.items():
                s_ = re.sub(rf"\b{k}\b", v, s_)
            result.add(s_)
        return tuple(result)

    exprs = list(filter(_is_expr_valid, solve_batch_eqns(relevant_updates, variable)))
    expr_templates = tuple(
        set(
            ExprTemplate.from_expr(proc_exs, context)
            for expr_str in exprs
            for proc_exs in _process_expr(expr_str)
        )
    )
    guards_ = tuple(_process_expr(g) for g in raw_template.guards if _is_expr_valid(g))
    guards = tuple(set(x for xs in guards_ for x in xs))

    return ConcreteExprGrammar(expr_templates, guards)


ExprType = TypeVar("ExprType")
RelSigType = TypeVar("RelSigType")
UnkMapType = TypeVar("UnkMapType")


class QEGrammarConstructor(Generic[ExprType, RelSigType, UnkMapType]):
    max_unroll_depth: int

    def __init__(
        self,
        max_unroll_depth: int,
        mode: Literal["complete", "partial"],
    ) -> None:
        super().__init__()
        self.max_unroll_depth = max_unroll_depth
        self.mode = mode

    def construct_grammar(
        self,
        prog: ExprType,
        rel_sig: RelSigType,
        unk_map: UnkMapType,
    ) -> dict[str, ConcreteExprGrammar]:
        raise NotImplementedError

    def encode_program(
        self: Self,
        prog: ExprType,
        rel_sig: RelSigType,
        unk_map: UnkMapType,
    ) -> Formula:
        raise NotImplementedError

    def extract_qe_formula(
        self: Self,
        f: Formula,
    ) -> RawTemplate:
        acc = RawTemplate()
        self._extract_qe_formula(f, acc)
        return acc

    def _extract_qe_formula(self: Self, f: Formula, acc: RawTemplate) -> None:
        match f:
            case PredicateApp(PredicateKinds.EQ, _):
                acc.updates.add(f)

            case PredicateApp():
                acc.guards.add(f)

            case Conjunction(fs) | Disjunction(fs):
                for f in fs:
                    self._extract_qe_formula(f, acc)

            case Negation(f):
                raise ValueError(f"Unsupported negation: {f}")


class IntermLangQEGrammarConstructor(
    QEGrammarConstructor[ir.Expr, RelationalSignature, dict[str, IRUnknownSpec]]
):
    Env: TypeAlias = frozendict[str, Expression]

    def __init__(
        self,
        max_unroll_depth: int,
        mode: Literal["complete", "partial"],
        stream_type: str,
    ) -> None:
        super().__init__(max_unroll_depth, mode)

        self.stream_type = stream_type

        self.fresh_var_ctr = 0
        self.fresh_var_map: dict[str, str] = {}

        # {name: {op: fresh_var}}
        self.uninterpreted_map: dict[str, dict[str, str]] = defaultdict(dict)

        # {fresh_var: full_expr}
        self.uninterpreted_to_expr: dict[str, str] = {}
        self.prelude = frozendict(
            log=VSymPreludeFunc(self.handle_log),
            exp=VSymPreludeFunc(self.handle_exp),
            abs=VSymPreludeFunc(self.handle_abs),
        )

    def _replace_expr(self, expr: Expression, func: str) -> ir.Value:
        """
        Add a fresh variable to the environment
        and return the name of the fresh variable

        name: the name of the variable
        expr_if_any: the expression of the variable if any (expr larger than single var)
        """

        expr_sp = to_sympy(expr).simplify()
        if expr_sp.is_constant():
            return ir.VNumber(value=float(expr_sp))

        match expr:
            case Variable(name=name):
                var_name = name
            case _:
                for k in self.fresh_var_map:
                    k_sp = parse_expr(k)
                    if expr_sp == k_sp:
                        var_name = self.fresh_var_map[k]
                        break
                else:
                    var_name = f"t{self.fresh_var_ctr}"
                    self.fresh_var_ctr += 1
                    self.fresh_var_map[str(expr_sp)] = var_name

        v = f"{func}_{var_name}"
        self.uninterpreted_to_expr[v] = f"{func}({var_name})"
        self.uninterpreted_map.setdefault(str(expr_sp), {})[func] = v
        return VSymbol(v)

    def handle_abs(self, v: ir.Value, env: Env) -> list[tuple[Formula, ir.Value]]:
        return [(ValueTop(), v)]

    def handle_log(self, v: ir.Value, env: Env) -> list[tuple[Formula, ir.Value]]:
        match v:
            case VSymbol(name=name):
                return [(ValueTop(), self._replace_expr(Variable(name), "log"))]

            # REDUCE doesn't handle log identities well
            # so we need to do the hack
            # rewrite log(x/y) to log(x) - log(y)
            case VSymbolExpr(expr=FunctionApp(FunctionKinds.DIV, (l, r))):
                return [
                    (
                        ValueTop(),
                        VSymbolExpr(
                            FunctionApp(
                                FunctionKinds.SUB,
                                (
                                    val_to_expr(self._replace_expr(l, "log")),
                                    val_to_expr(self._replace_expr(r, "log")),
                                ),
                            )
                        ),
                    )
                ]

            case VSymbolExpr(expr=expr):
                return [(ValueTop(), self._replace_expr(expr, "log"))]

            case _:
                raise ValueError(f"Unsupported value: {v}")

    def handle_exp(self, v: ir.Value, env: Env) -> list[tuple[Formula, ir.Value]]:
        match v:
            case VSymbol(name=name):
                return [
                    (
                        ValueTop(),
                        VSymbolExpr(
                            FunctionApp(
                                FunctionKinds.POW, (Variable("e"), Variable(name))
                            )
                        ),
                    )
                ]

            # REDUCE doesn't handle nonatomic formulae
            # so we need to do the hack
            # rewrite exp(x+y) to exp(x) * exp(y)
            case VSymbolExpr(expr=FunctionApp(FunctionKinds.ADD, (l, r))) | VSymbolExpr(
                expr=FunctionApp(FunctionKinds.SUB, (l, r))
            ):
                vl = self._replace_expr(l, "exp")
                vr = self._replace_expr(
                    r
                    if v.expr == FunctionKinds.ADD
                    else FunctionApp(FunctionKinds.NEG, (r,)),
                    "exp",
                )

                el = val_to_expr(vl)
                er = val_to_expr(vr)
                return [
                    (
                        ValueTop(),
                        VSymbolExpr(
                            FunctionApp(
                                FunctionKinds.MUL,
                                (el, er),
                            )
                        ),
                    )
                ]

            case VSymbolExpr(expr=expr):
                return [(ValueTop(), self._replace_expr(expr, "exp"))]

            case _:
                raise ValueError(f"Unsupported value: {v}")

    def _construct_symbolic_input(
        self,
    ) -> tuple[list[VSymbol] | list[ir.VPair], list[VSymbol] | list[ir.VPair]]:
        is_scalar = not self.stream_type.startswith("(")
        if is_scalar:
            symbolic_stream = [
                VSymbol(f"x{i}") for i in range(self.max_unroll_depth - 1)
            ]
            symbolic_stream_with_current = symbolic_stream + [
                VSymbol(VAR_NAME.CURRENT_ELEMENT)
            ]
            return (symbolic_stream, symbolic_stream_with_current)

        match self.stream_type:
            case "(float, float)":
                symbolic_stream_ = [
                    ir.VPair((VSymbol(f"x{i}"), VSymbol(f"y{i}")))
                    for i in range(self.max_unroll_depth - 1)
                ]
                symbolic_stream_with_current_ = symbolic_stream_ + [
                    ir.VPair(
                        (
                            VSymbol(VAR_NAME.CURRENT_ELEMENT_1),
                            VSymbol(VAR_NAME.CURRENT_ELEMENT_2),
                        )
                    )
                ]
                return (symbolic_stream_, symbolic_stream_with_current_)

        raise NotImplementedError("TBD")

    def construct_grammar(
        self,
        prog: Expr,
        rel_sig: RelationalSignature,
        unk_map: dict[str, IRUnknownSpec],
    ) -> dict[str, ConcreteExprGrammar]:
        try:
            rel_f, unk_fs = self.encode_program_sep(prog, rel_sig, unk_map)
        except NotImplementedError:
            logging.warning("Failed to encode program, fallback to search")
            return {}

        unrolled_params = [f"x{i}" for i in range(self.max_unroll_depth - 1)] + [
            f"y{i}" for i in range(self.max_unroll_depth - 1)
        ]
        fresh_params = [
            op for up in unrolled_params for op in self.uninterpreted_map[up].values()
        ]

        t = RawTemplate()
        for unk_f in unk_fs:
            encoded_f = simplify(Conjunction.create(rel_f, unk_f))
            f_str = to_reduce_formula(encoded_f)
            try:
                qe_f = run_rlqe(f_str, unrolled_params + fresh_params)
                t.update(self.extract_qe_formula(qe_f))
            except TimeoutError:
                assert isinstance(rel_f, Conjunction)
                other_exprs = tuple(
                    e
                    for e in rel_f.exprs
                    if "prev_sq_s_x" not in str(e) and "prev_sq_s_y" not in str(e)
                )  # FIXME: replace this hack with proper dependency analysis

                f_str = to_reduce_formula(
                    simplify(Conjunction.create(*other_exprs, unk_f))
                )
                try:
                    qe_f = run_rlqe(f_str, unrolled_params + fresh_params, 5)
                    t.update(self.extract_qe_formula(qe_f))
                except TimeoutError:
                    continue

        assert isinstance(prog, ir.ELam), "input program is not a lambda"
        aux_vars = set(prog.params) - {"xs"}

        expr_to_fresh_var = {
            expr: fv for expr, fv in self.fresh_var_map.items() if expr != fv
        }
        fresh_var_to_updated_exprs = {}

        rel_f_str = to_reduce_formula(rel_f)
        # TODO: potential ways to speed up
        # 1. use z3 instead of REDUCE
        # 2. parallelize
        for expr, fv in tqdm(expr_to_fresh_var.items(), desc="Solving fresh vars"):
            fv_f_str = f"{fv} = ({expr})"
            qe_fv_f = run_rlqe(
                f"({fv_f_str}) and ({rel_f_str})", unrolled_params + fresh_params
            )
            fv_t = self.extract_qe_formula(qe_fv_f)

            solved_exprs = _raw_template_to_template(
                fv_t, fv, self.max_unroll_depth, rel_sig, aux_vars, None
            ).exprs
            solved_exprs_ = tuple(
                expr_t.expr.format_map(
                    {k: v for k, ((_, v),) in expr_t.ctx_ios.items()}
                )
                for expr_t in solved_exprs
            )
            fresh_var_to_updated_exprs[fv] = solved_exprs_

        expr_templates = {}
        for unk in unk_map:
            expr_templates[unk] = _raw_template_to_template(
                t,
                unk,
                self.max_unroll_depth,
                rel_sig,
                aux_vars,
                self.uninterpreted_to_expr,
                fresh_var_to_updated_exprs,
            )
        return expr_templates

    def encode_program_sep(
        self: Self,
        prog: Expr,
        rel_sig: RelationalSignature,
        unk_map: dict[str, IRUnknownSpec],
    ) -> tuple[Formula, tuple[Formula, ...]]:
        assert isinstance(prog, ir.ELam), "input program is not a lambda"
        aux_vars = set(prog.params) - {"xs"}

        def _encode_equality(var: str, expr: Expression) -> PredicateApp:
            sp_expr = to_sympy_expr(expr)
            num, deno = sp.fraction(sp.together(sp_expr))

            if not _expression_supported_by_reduce(expr):
                return PredicateApp(PredicateKinds.EQ, (Variable(var), Variable(var)))

            if deno.is_number:
                return PredicateApp(
                    PredicateKinds.EQ,
                    (Variable(var), expr),
                )

            return PredicateApp(
                PredicateKinds.EQ,
                (
                    FunctionApp(
                        FunctionKinds.MUL,
                        (Variable(var), sympy_expr_to_expression(deno)),
                    ),
                    sympy_expr_to_expression(num),
                ),
            )

        def _encode_paths(
            exprs: Sequence[RelationalSignatureExpr],
            expr_names: Sequence[str],
            env: Env,
        ) -> tuple[Formula, ...]:
            fs = []

            for expr, ex_name in zip(exprs, expr_names):
                loc_f: Formula = ValueBottom()
                visited_cond = set()
                for cond, val in sym_exec(expr.expand(), env):
                    cond = simplify(cond)
                    if cond in visited_cond:
                        continue
                    visited_cond.add(cond)

                    if not check_sat(cond):
                        continue

                    loc_f = Disjunction.create(
                        loc_f,
                        Conjunction.create(
                            cond,
                            _encode_equality(ex_name, val_to_expr(val, expr.map_idx)),
                        ),
                    )

                fs.append(loc_f)
            return tuple(fs)

        (
            symbolic_stream_,
            symbolic_stream_with_current_,
        ) = self._construct_symbolic_input()
        symbolic_stream = to_value_list(symbolic_stream_)
        symbolic_stream_with_current = to_value_list(symbolic_stream_with_current_)

        rel_sig_formula = _encode_paths(
            tuple(rel_sig.values()),
            tuple(rel_sig),
            frozendict(
                {
                    **{v: VSymbol(v) for v in aux_vars},
                    VAR_NAME.INPUT_STREAM: symbolic_stream,
                    **self.prelude,
                }
            ),
        )
        unk_formula = _encode_paths(
            tuple(
                (
                    map_expr_to_partial_stream(v.equivalent_expr)
                    if self.mode == "partial"
                    else v.equivalent_expr
                )
                for v in unk_map.values()
            ),
            tuple(unk_map),
            frozendict(
                {
                    **{v: VSymbol(v) for v in aux_vars},
                    VAR_NAME.INPUT_STREAM: symbolic_stream_with_current,
                    "xs'": symbolic_stream,
                    **self.prelude,
                }
            ),
        )

        return (Conjunction.create(*rel_sig_formula), unk_formula)


class ImperativeQEGrammarConstructor(
    QEGrammarConstructor[
        ast.FunctionDef, ImperativeRelationalSignature, Sequence[ASTUnknownSpec]
    ]
):
    def construct_grammar(
        self,
        prog: ast.FunctionDef,
        rel_sig: ImperativeRelationalSignature,
        unk_map: Sequence[ASTUnknownSpec],
    ) -> dict[str, ConcreteExprGrammar]:
        f_str = to_reduce_formula(self.encode_program(prog, rel_sig, unk_map))
        unrolled_params = [f"xs{i}" for i in range(self.max_unroll_depth - 1)]

        qe_f = run_rlqe(f_str, unrolled_params)
        t = self.extract_qe_formula(qe_f)

        expr_templates = {}
        for unk in unk_map:
            match unk:
                case ImperativeUnknownSpec(variable=v):
                    expr_templates[v] = _raw_template_to_template(
                        t, v, self.max_unroll_depth, rel_sig, set()
                    )
                case UnknownSpec(unk_id=u):
                    expr_templates[u] = _raw_template_to_template(
                        t, u, self.max_unroll_depth, rel_sig, set()
                    )
        return expr_templates

    def encode_program(
        self: Self,
        prog: ast.FunctionDef,
        rel_sig: ImperativeRelationalSignature,
        unk_map: Sequence[ASTUnknownSpec],
    ) -> Formula:
        f_rel_sig = self.qe_encode_rel_sig_formula(
            unroll_loop(prog, self.max_unroll_depth - 1),
            rel_sig,
            self.max_unroll_depth - 1,
        )
        f_unk_map = [
            ImperativeQEGrammarConstructor.qe_encode_subprogram_imperative(
                unroll_loop(prog, self.max_unroll_depth, VAR_NAME.CURRENT_ELEMENT),
                self.max_unroll_depth,
                unk.variable if isinstance(unk, ImperativeUnknownSpec) else unk.unk_id,
                unk.equivalent_expr,
                unk.unk_id,
            )
            for unk in unk_map
        ]

        return Conjunction.create(f_rel_sig, *f_unk_map)

    @staticmethod
    def qe_encode_rel_sig_formula(
        unrolled_func: ast.FunctionDef,
        rel_sig: ImperativeRelationalSignature,
        unroll_depth: int,
    ) -> Formula:
        # assumption: all locals are immutable except the loop variables

        formula_mapping: dict[str, Formula] = {}

        for param, expr_str in rel_sig.items():
            expr: ast.expr
            if expr_str == VAR_NAME.RET_TEMP_VAR:
                expr = ast.Name(id=OUT_KEYWORD)
            else:
                expr = parse_term(expr_str)

            formula_mapping[
                param
            ] = ImperativeQEGrammarConstructor.qe_encode_subprogram_imperative(
                unrolled_func, unroll_depth, param, expr, "n/a"
            )

        return Conjunction.create(*formula_mapping.values())

    @staticmethod
    def compute_wp(
        stmt_or_stmts: ast.stmt | list[ast.stmt], post_condition: Formula
    ) -> Formula:
        match stmt_or_stmts:
            case stmts if isinstance(stmts, list):
                return ImperativeQEGrammarConstructor.compute_wp_seq(
                    stmts, post_condition
                )
            case stmt if isinstance(stmt, ast.stmt):
                return ImperativeQEGrammarConstructor.compute_wp_stmt(
                    stmt, post_condition
                )

        raise ValueError(f"Invalid stmt_or_stmts: {stmt_or_stmts}")

    @staticmethod
    def compute_wp_stmt(
        stmt: ast.stmt,
        post_condition: Formula,
    ) -> Formula:
        match stmt:
            case ast.Assign(targets=targets, value=value):
                assert len(targets) == 1, f"Unsupported targets: {targets}"
                target_expr = to_expression(targets[0])
                value_expr = to_expression(value)
                return substitute_variable(post_condition, target_expr, value_expr)

            case ast.If(test=test, body=body, orelse=orelse):
                cond_formula = to_formula(test)
                return Conjunction.create(
                    Implication(
                        cond_formula,
                        ImperativeQEGrammarConstructor.compute_wp_seq(
                            body, post_condition
                        ),
                    ),
                    Implication(
                        Negation(cond_formula),
                        ImperativeQEGrammarConstructor.compute_wp_seq(
                            orelse, post_condition
                        ),
                    ),
                )

            case ast.AugAssign(target=target, op=op, value=value):
                target_expr = to_expression(target)
                value_expr = to_expression(ast.BinOp(left=target, op=op, right=value))
                return substitute_variable(post_condition, target_expr, value_expr)

        raise ValueError(f"Unsupported stmt: {ast.unparse(stmt)}")

    @staticmethod
    def compute_wp_seq(stmts: list[ast.stmt], post_condition: Formula) -> Formula:
        return foldr(
            ImperativeQEGrammarConstructor.compute_wp_stmt, post_condition, stmts
        )

    @staticmethod
    def _replace_special_call(call: ast.Call, unroll_depth: int) -> ast.expr | None:
        match call:
            case ast.Call(
                func=ast.Name(id="len"), args=[ast.Name(id=VAR_NAME.INPUT_STREAM)]
            ):
                return ast.Constant(value=unroll_depth)

        return None

    @staticmethod
    def qe_encode_subprogram_imperative(
        unrolled_func: ast.FunctionDef,
        unroll_depth: int,
        var_name: str,
        expr: ast.expr | ast.stmt,
        unk_id: str,
    ) -> Formula:
        match expr:
            case ast.Name(id=name):
                need_replace = False
                if var_name == name:
                    var_name = f"{unk_id}_{name}"
                    need_replace = True
                f = ImperativeQEGrammarConstructor.compute_wp_seq(
                    unrolled_func.body,
                    PredicateApp(
                        PredicateKinds.EQ, (Variable(var_name), Variable(name))
                    ),
                )

                return (
                    substitute_variable(f, Variable(var_name), Variable(name))
                    if need_replace
                    else f
                )

            case ast.Constant(value=value):
                return PredicateApp(
                    PredicateKinds.EQ, (Variable(var_name), Constant(value))
                )

            case ast.Call() if (
                new_expr := ImperativeQEGrammarConstructor._replace_special_call(
                    expr, unroll_depth
                )
            ):
                return ImperativeQEGrammarConstructor.qe_encode_subprogram_imperative(
                    unrolled_func, unroll_depth, var_name, new_expr, unk_id
                )

            case ast.For(iter=ast.Name(id=VAR_NAME.INPUT_STREAM)):
                left_vars = extract_left_values(expr)
                assert (
                    var_name in left_vars
                ), f"Variable {var_name} not found in loop of {left_vars}"

                return ImperativeQEGrammarConstructor.qe_encode_subprogram_imperative(
                    unrolled_func, unroll_depth, var_name, ast.Name(id=var_name), unk_id
                )

            case ast.BinOp() | ast.AugAssign() | ast.UnaryOp() | ast.Compare():
                return ImperativeQEGrammarConstructor.compute_wp_seq(
                    unrolled_func.body,
                    PredicateApp(
                        PredicateKinds.EQ, (Variable(var_name), to_expression(expr))
                    ),
                )

        raise ValueError(f"Unsupported expr: {ast.unparse(expr)}")
