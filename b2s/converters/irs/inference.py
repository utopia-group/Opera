import ast

from frozendict import frozendict

from b2s.const import VAR_NAME
from b2s.lang import (
    EBinOp,
    ECall,
    EIte,
    ELam,
    ELet,
    EMapGet,
    EMapNil,
    EMapUpdate,
    ENil,
    EPair,
    EPythonExpr,
    EStream,
    EUnknown,
    EVar,
    Expr,
    fold_expr,
    fold_expr_rec,
)
from b2s.lang_interp import prelude
from b2s.rfs import (
    ImperativeRelationalSignature,
    RelationalSignature,
    RelationalSignatureExpr,
    RSExprEnv,
)
from b2s.sketch_types import IRSketch, IRUnknownSpec
from b2s.utils import extract_left_values, throw_

RESERVED_VAR_NAMES = {"len"}


def get_smallest_stream_exprs(
    expr: Expr, tld_lets: RSExprEnv, cur_let_name: str | None = None
) -> dict[Expr, tuple[str | None, RSExprEnv]]:
    """
    Extract the smallest stream expressions from the given expression.

    :param expr: the expression to extract from
    :param tld_lets: the let bindings in the current scope; used to expand the
        extracted expressions
    :param cur_let_name: the name of the current let binding; used to assign
        names to the extracted expressions
    :return: a dictionary mapping the extracted expressions to a tuple of
        (1) a potential name for the expression if exists, or None otherwise,
        and (2) the let bindings in the current scope
    """

    match expr:
        case ECall(func, args):
            exprs = {}
            if any(isinstance(arg, EStream) for arg in args):
                # the name defaults to the current let binding name
                exprs[expr] = (cur_let_name, tld_lets)
                match func:
                    case EVar(name) if name not in prelude or name == "len":
                        exprs[expr] = (name, tld_lets)
            return (
                exprs
                | get_smallest_stream_exprs(func, tld_lets)
                | {
                    k: v
                    for arg in args
                    for k, v in get_smallest_stream_exprs(arg, tld_lets).items()
                }
            )
        case ELam(_, body):
            return get_smallest_stream_exprs(body, tld_lets)
        case ELet(name, expr, body):
            let_expr_pairs = get_smallest_stream_exprs(
                expr, tld_lets, name if name != "S" else None
            )
            new_tld = tuple((*tld_lets, (name, expr)))
            if len(let_expr_pairs) == 1:
                let_expr, orig_name = let_expr_pairs.popitem()
                return {let_expr: orig_name or name} | get_smallest_stream_exprs(
                    body, new_tld
                )
            return let_expr_pairs | get_smallest_stream_exprs(body, new_tld)
        case ECall(func, args):
            return get_smallest_stream_exprs(func, tld_lets) | {
                k: v
                for arg in args
                for k, v in get_smallest_stream_exprs(arg, tld_lets).items()
            }
        case EBinOp(left, _, right):
            return get_smallest_stream_exprs(
                left, tld_lets
            ) | get_smallest_stream_exprs(right, tld_lets)
        case EIte(cond, then, els):
            return (
                get_smallest_stream_exprs(cond, tld_lets)
                | get_smallest_stream_exprs(then, tld_lets)
                | get_smallest_stream_exprs(els, tld_lets)
            )
        case EPair(elts):
            return {
                k: v
                for elt in elts
                for k, v in get_smallest_stream_exprs(elt, tld_lets).items()
            }
        case EMapUpdate(expr, updates):
            return get_smallest_stream_exprs(expr, tld_lets) | {
                k: v
                for update in updates.values()
                for k, v in get_smallest_stream_exprs(update, tld_lets).items()
            }
        case EMapGet(expr, _):
            return get_smallest_stream_exprs(expr, tld_lets)
        case _:
            return {}


def infer_relational_signature(expr: ELam) -> RelationalSignature:
    rel_sig = RelationalSignature(
        {
            VAR_NAME.PREVIOUS_OUTPUT: RelationalSignatureExpr(expr.body),
        }
    )

    fresh_var_counter = 0

    stream_exprs = get_smallest_stream_exprs(expr.body, ())
    for stream_expr, (name, lets) in stream_exprs.items():
        match stream_expr:
            case ECall(EVar("foldl"), (ELam(_, EMapUpdate(_, updates)), _, _)):
                for k in updates:
                    rel_sig[f"prev_{k}"] = RelationalSignatureExpr(stream_expr, lets, k)
                continue

        if name is None:
            name = f"var{fresh_var_counter}"
            fresh_var_counter += 1

        rel_sig[f"prev_{name}"] = RelationalSignatureExpr(stream_expr, lets)
    return rel_sig


def sketchify(
    expr: ELam,
    rel_sig: RelationalSignature,
    ablation_no_decomp: bool,
    partial_mode: bool,
) -> IRSketch:
    smallest_stream_exprs = {
        k: v for k, v in rel_sig.items() if k not in {VAR_NAME.PREVIOUS_OUTPUT}
    }

    prog_used_len = fold_expr(
        expr,
        f_var=lambda v: "len" == v.name,
        f_stream=lambda _: False,
        f_int=lambda _: False,
        f_bool=lambda _: False,
        f_str=lambda _: False,
        f_binop=lambda left, _, right: left or right,
        f_call=lambda func, args: func or any(args),
        f_lam=lambda params, body: body,
        f_let=lambda name, expr, body: expr or body,
        f_ite=lambda cond, then, els: cond or then or els,
        f_nil=lambda: False,
        f_pair=lambda elts: any(elts),
        f_unk=lambda _: False,
        f_map=lambda _, exprs: any(exprs.values()),
        f_map_nil=lambda: False,
        f_map_get=lambda ex, _: ex,
        f_python_expr=lambda _: False,
    )
    if not prog_used_len:
        smallest_stream_exprs.pop(VAR_NAME.PREVIOUS_LENGTH, None)
        rel_sig.pop(VAR_NAME.PREVIOUS_LENGTH, None)

    sketch_body = expr.body

    rse_to_unk_map: dict[RelationalSignatureExpr, EUnknown] = {
        v: EUnknown(k.replace("prev_", "new_"))
        for k, v in smallest_stream_exprs.items()
    }
    unk_to_rse_map: dict[str, RelationalSignatureExpr] = {
        v.unk_id: k for k, v in rse_to_unk_map.items()
    }
    unk_update_map_to_exp: dict[EMapUpdate, Expr] = {}

    def _revert_unks(expr: Expr) -> Expr:
        return fold_expr(
            expr,
            f_var=lambda e: e,
            f_stream=lambda e: e,
            f_int=lambda e: e,
            f_bool=lambda e: e,
            f_str=lambda e: e,
            f_binop=lambda left, op, right: EBinOp(left, op, right),
            f_call=lambda func, args: ECall(func, args),
            f_lam=lambda params, body: ELam(params, body),
            f_let=lambda name, expr, body: ELet(name, expr, body),
            f_ite=lambda cond, then, els: EIte(cond, then, els),
            f_nil=lambda: ENil(),
            f_pair=lambda elts: EPair(elts),
            f_unk=lambda unk_id: unk_to_rse_map[unk_id].expr
            if unk_to_rse_map[unk_id].map_idx
            is None  # if map_idx is set, then it's a map update we'll cover in f_map
            else EUnknown(unk_id),
            f_map=lambda e, updates: unk_update_map_to_exp.get(
                EMapUpdate(e, updates), EMapUpdate(e, updates)
            ),
            f_map_nil=lambda: EMapNil(),
            f_map_get=lambda e, k: EMapGet(e, k),
            f_python_expr=lambda e: EPythonExpr(e),
        )

    def id1(_, x):
        return x

    def _replace_call(_, func: Expr, args: tuple[Expr, ...]) -> Expr:
        args_mapped = tuple(_revert_unks(arg) for arg in args)
        call = ECall(func, args)
        orig_call = ECall(func, args_mapped)

        rse = next(filter(lambda x: x.expr == orig_call, rse_to_unk_map), None)
        if rse is None:
            return call

        if rse.map_idx is None:
            result_expr = rse_to_unk_map[rse]
            match args:
                case (ELam(_, body), _, _) if partial_mode:
                    return ELet(VAR_NAME.STATE_MAP, result_expr, body)  # type: ignore
                case _:
                    return result_expr

        match args:
            case (ELam(_, EMapUpdate(state, orig_updates)), _, _):
                updates = frozendict(
                    {
                        k: next(filter(lambda p: p[0].expr == orig_call and p[0].map_idx == k, rse_to_unk_map.items()))[1]  # type: ignore
                        for k in orig_updates
                    }
                )

            case _:
                assert False, f"map update not found in {args}"

        result_expr = EMapUpdate(state, updates)  # type: ignore
        # instead of ?? = foldl(f, ...), we have f(??) = foldl(f, ...)
        # this reuses expressions in the original program
        if partial_mode:
            assert isinstance(state, EVar)
            result_expr = ELet(state.name, result_expr, EMapUpdate(state, orig_updates))  # type: ignore

        unk_update_map_to_exp[result_expr] = orig_call
        return result_expr

    def _tweak_let(f, name: str, expr: Expr, body: Expr) -> Expr:
        let_exp = ELet(name, expr, body)
        if (
            name == "S"
            and isinstance(expr, EMapUpdate)
            and expr in unk_update_map_to_exp
        ):
            orig_prog = unk_update_map_to_exp[expr]
            match orig_prog:
                case ECall(EVar("foldl"), (_, ex, _)) if isinstance(ex, EMapUpdate):
                    # TODO: add some checks here
                    return ELet("S", f(ex), let_exp)
        return let_exp

    sketch_body = fold_expr_rec(
        sketch_body,
        f_var=id1,
        f_stream=id1,
        f_int=id1,
        f_bool=id1,
        f_str=id1,
        f_binop=lambda _, left, op, right: EBinOp(left, op, right),
        f_lam=lambda _, params, body: ELam(params, body),
        f_let=_tweak_let,
        f_ite=lambda _, cond, then, els: EIte(cond, then, els),
        f_nil=lambda _: ENil(),
        f_pair=lambda _, elts: EPair(elts),
        f_unk=lambda _, unk_id: throw_(Exception(f"unknown {unk_id}")),
        f_call=_replace_call,
        f_map=lambda _, ex, updates: EMapUpdate(ex, updates),
        f_map_get=lambda _, e, k: EMapGet(e, k),
        f_map_nil=lambda _: EMapNil(),
        f_python_expr=lambda _, e: EPythonExpr(e),
    )

    if ablation_no_decomp:
        prog_out_rse = next(iter(rel_sig.values()))
        prog_out_unk = EUnknown(VAR_NAME.CURRENT_PROG_OUT)
        sketch_body = EPair.create(prog_out_unk, EPair.create(*rse_to_unk_map.values()))
        sketch = ELam((*rel_sig.keys(), VAR_NAME.CURRENT_ELEMENT), sketch_body)

        return IRSketch(
            sketch,
            {
                prog_out_unk.unk_id: IRUnknownSpec(
                    prog_out_unk.unk_id, prog_out_rse, None
                ),
                **{
                    unk_id: IRUnknownSpec(unk_id, rse, None)
                    for unk_id, rse in unk_to_rse_map.items()
                },
            },
        )

    if not partial_mode:
        sketch_body = EPair.create(sketch_body, EPair.create(*rse_to_unk_map.values()))
    else:
        sketch_body = EPair.create(
            sketch_body,
            EPair.create(
                *[
                    rse_to_unk_map[rse]
                    if rse.map_idx is None
                    else EMapGet(EVar(VAR_NAME.STATE_MAP), rse.map_idx)
                    for rse in rse_to_unk_map
                ]
            ),
        )
    sketch = ELam((*rel_sig.keys(), VAR_NAME.CURRENT_ELEMENT), sketch_body)

    return IRSketch(
        sketch,
        {
            unk_id: IRUnknownSpec(unk_id, rse, None)
            for unk_id, rse in unk_to_rse_map.items()
        },
    )


def infer_imperative_relational_signature(
    func: ast.FunctionDef,
) -> ImperativeRelationalSignature:
    rel_sig = ImperativeRelationalSignature(
        {
            VAR_NAME.PREVIOUS_OUTPUT: VAR_NAME.RET_TEMP_VAR,
            VAR_NAME.PREVIOUS_LENGTH: "len(xs)",
        }
    )
    for node in ast.walk(func):
        match node:
            case ast.For(body=body):
                left_vars = [v for b in body for v in extract_left_values(b)]
                rel_sig.update({f"prev_{v}": v for v in left_vars})

            case ast.Call(func=call_func, args=args):
                for arg in args:
                    match arg:
                        case ast.Name(VAR_NAME.INPUT_STREAM):
                            break
                else:
                    continue

                call_func_str = ast.unparse(call_func)
                call_expr_str = ast.unparse(node)
                var_name = (
                    f"prev_{call_func_str}".replace("(", "_")
                    .replace(")", "")
                    .replace(",", "_")
                )
                assert (
                    var_name not in rel_sig or rel_sig[var_name] == call_expr_str
                ), f"{var_name} already in {rel_sig}"
                rel_sig[var_name] = call_expr_str

            case ast.Assign(targets, value):
                # TODO: auxiliary variables may be used in the future
                pass

    return rel_sig
