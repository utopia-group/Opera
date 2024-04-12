from dataclasses import dataclass
from itertools import product

from frozendict import frozendict

from b2s.lang import *
from b2s.synthesizers.qe_types import *


@dataclass(frozen=True, slots=True, eq=True)
class VSymbol(Value):
    name: Id

    def _handle_binop(self, other: Value, op_kind: BinOpKinds) -> Value:
        return VSymbolExpr(
            FunctionApp(
                func=IR_TO_FUNCTION_KINDS[op_kind],
                args=(Variable(self.name), val_to_expr(other)),
            )
        )


@dataclass(frozen=True, slots=True, eq=True)
class VSymbolExpr(Value):
    expr: Expression

    def _handle_binop(self, other: Value, op_kind: BinOpKinds) -> Value:
        return VSymbolExpr(
            FunctionApp(
                func=IR_TO_FUNCTION_KINDS[op_kind],
                args=(self.expr, val_to_expr(other)),
            )
        )

    def __add__(self, other: Value) -> Value:
        return self._handle_binop(other, BinOpKinds.ADD)

    def __sub__(self, other: Value) -> Value:
        return self._handle_binop(other, BinOpKinds.SUB)

    def __mul__(self, other: Value) -> Value:
        return self._handle_binop(other, BinOpKinds.MUL)

    def __truediv__(self, other: Value) -> Value:
        return self._handle_binop(other, BinOpKinds.DIV)

    def __pow__(self, other: Value) -> Value:
        return self._handle_binop(other, BinOpKinds.POW)


@dataclass(frozen=True, slots=True, eq=True)
class VSymbolFormula(Value):
    formula: Formula


@dataclass(frozen=True, slots=True, eq=True)
class VSymPreludeFunc(Value):
    func: Callable[..., list[tuple[Formula, Value]]]


def val_to_expr(val: Value, map_idx: str | None = None) -> Expression:
    match val:
        case VSymbol(name):
            return Variable(name)
        case VNumber(n) | VBool(n) | VString(n):
            return Constant(n)
        case VSymbolExpr(expr):
            return expr
        case VMap(value):
            assert len(value) == 1 or map_idx in value, "map has more than one key"
            return val_to_expr(value.get(map_idx, next(iter(value.values()))))  # type: ignore

    assert False, f"Unsupported value: {val}"


def prelude_foldl(
    func: Value, init: Value, foldable: Value, env: Env
) -> list[tuple[Formula, Value]]:
    states: list[tuple[Formula, Value, Value]] = [(ValueTop(), init, foldable)]
    results = []

    while states:
        cond, acc, node = states.pop()
        match node:
            case VNil():
                results.append((cond, acc))
            case VCons(value, nxt):
                for c, v in apply_function_values(func, (acc, value), env):
                    states.append((Conjunction.create(cond, c), v, nxt))
            case _:
                results.append((cond, VError("lst input to foldl is not a list")))

    return results


def prelude_head(lst: Value, env: Env) -> list[tuple[Formula, Value]]:
    match lst:
        case VNil():
            return [(ValueTop(), VError("empty list does not have a head"))]
        case VCons(value, _):
            return [(ValueTop(), value)]
        case _:
            return [(ValueTop(), VError("lst input to head is not a list"))]


def prelude_length(lst: Value, env: Env) -> list[tuple[Formula, Value]]:
    states: list[tuple[Formula, int, Value]] = [(ValueTop(), 0, lst)]
    results = []

    while states:
        cond, acc, node = states.pop()
        match node:
            case VNil():
                results.append((cond, VNumber(acc)))
            case VCons(_, nxt):
                states.append((ValueTop(), acc + 1, nxt))
            case _:
                results.append((cond, VError("lst input to foldl is not a list")))

    return results


def prelude_fst(pair: Value, env: Env) -> list[tuple[Formula, Value]]:
    match pair:
        case VPair(elts=elts):
            return [(ValueTop(), elts[0])]
        case _:
            return [(ValueTop(), VError("input to fst is not a pair"))]

def prelude_snd(pair: Value, env: Env) -> list[tuple[Formula, Value]]:
    match pair:
        case VPair(elts=elts):
            return [(ValueTop(), elts[1])]
        case _:
            return [(ValueTop(), VError("input to snd is not a pair"))]


sym_prelude: Env = frozendict(
    {
        "foldl": VSymPreludeFunc(prelude_foldl),
        "head": VSymPreludeFunc(prelude_head),
        "len": VSymPreludeFunc(prelude_length),
        "fst": VSymPreludeFunc(prelude_fst),
        "snd": VSymPreludeFunc(prelude_snd),
    }
)  # type: ignore


def apply_function_values(
    func: Value, args: tuple[Value, ...], env: Env
) -> list[tuple[Formula, Value]]:
    match func:
        case VClosure(params, body, new_env):
            assert len(params) == len(args), "Wrong number of arguments"
            new_env = new_env | frozendict(zip(params, args))
            return sym_exec(body, new_env)
        case VSymPreludeFunc(f):
            return f(*args, env)
        case _:
            assert False, f"Unsupported function: {func}"


def sym_exec(expr: Expr, env: Env) -> list[tuple[Formula, Value]]:
    results = []
    match expr:
        case EVar(x) | EStream(x):
            results.append(
                (
                    ValueTop(),
                    env.get(x, sym_prelude.get(x, VError(f"{x} is not defined"))),
                )
            )
        case ENumber(n) | EBool(n) | EString(n):
            results.append((ValueTop(), VSymbolExpr(Constant(n))))
        case EBinOp(left, op, right):
            left_exprs = sym_exec(left, env)
            right_exprs = sym_exec(right, env)

            for (cond_l, val_l), (cond_r, val_r) in product(left_exprs, right_exprs):
                cond = Conjunction.create(cond_l, cond_r)
                args = (val_to_expr(val_l), val_to_expr(val_r))
                func = IR_TO_FUNCTION_KINDS.get(op, None)
                if func is not None:
                    try:
                        f_expr = FunctionApp(func=func, args=args)
                        results.append((cond, VSymbolExpr(f_expr)))
                    except ZeroDivisionError:
                        continue
                    continue
                func = IR_TO_PREDICATE_KINDS.get(op, None)
                if func is not None:
                    results.append(
                        (
                            cond,
                            VSymbolFormula(
                                PredicateApp(
                                    pred=func,
                                    args=args,
                                )
                            ),
                        )
                    )
                    continue

                match op:
                    case BinOpKinds.CONS, _, _:
                        results.append((cond, VCons(val_l, val_r)))

                assert False, f"Unsupported binop: {op}"
        case ir.ECall(func, args):
            func_val_list = sym_exec(func, env)
            args_list = list(map(lambda arg: sym_exec(arg, env), args))

            for func_val_arg_pair in product(func_val_list, *args_list):
                assert len(func_val_arg_pair) == len(args) + 1
                cond = Conjunction.create(*tuple(cond for cond, _ in func_val_arg_pair))
                func_val = func_val_arg_pair[0][1]
                args_val = tuple(arg for _, arg in func_val_arg_pair[1:])  # type: ignore

                results.extend(
                    [
                        (Conjunction.create(cond, cond_l), val)
                        for cond_l, val in apply_function_values(
                            func_val, args_val, env
                        )
                    ]
                )
        case ir.ELam(params, body):
            results.append(
                (
                    ValueTop(),
                    VClosure(
                        params,
                        body,
                        frozendict({k: v for k, v in env.items() if k not in params}),
                    ),
                )
            )
        case ir.ELet(name, expr, body):
            for cond, val in sym_exec(expr, env):
                results.extend(
                    [
                        (Conjunction.create(body_cond, cond), body_val)
                        for body_cond, body_val in sym_exec(
                            body, frozendict({**env, name: val})
                        )
                    ]
                )
        case ir.EIte(cond, then, els):
            f_conds = sym_exec(cond, env)
            f_thens = sym_exec(then, env)
            f_elses = sym_exec(els, env)
            for (
                (cond_cond, cond_val),
                (then_cond, then_val),
                (els_cond, els_val),
            ) in product(f_conds, f_thens, f_elses):
                assert isinstance(cond_val, VSymbolFormula)
                results.append(
                    (
                        Conjunction.create(cond_cond, cond_val.formula, then_cond),
                        then_val,
                    )
                )
                results.append(
                    (
                        Conjunction.create(
                            cond_cond, Negation(cond_val.formula), els_cond
                        ),
                        els_val,
                    )
                )
        case ir.EMapNil():
            results.append((ValueTop(), VMap(frozendict())))
        case ir.ENil():
            results.append((ValueTop(), VNil()))
        case ir.EMapUpdate(e, updates):
            current_states = sym_exec(e, env)
            next_states = []

            for update_key in updates:
                for cond, val in current_states:
                    assert isinstance(val, VMap), "state is not a map"

                    state_dict = dict(val.value)
                    next_states.extend(
                        (
                            Conjunction.create(cond, c),
                            VMap(frozendict({**state_dict, update_key: v})),
                        )
                        for c, v in sym_exec(
                            updates[update_key], frozendict({**env, **state_dict})
                        )
                    )
                current_states.clear()
                current_states.extend(next_states)
                next_states.clear()

            results.extend(current_states)

        case ir.EMapGet(e, k):
            e_vals = sym_exec(e, env)
            for cond, val in e_vals:
                assert isinstance(val, VMap), "state is not a map"
                results.append((cond, val.value[k]))
        
        case _:
            raise NotImplementedError(f"Unsupported expr: {expr}")

    return list(filter(lambda p: check_sat(p[0]), results))
