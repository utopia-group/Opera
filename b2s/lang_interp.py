import math
from collections import defaultdict

from frozendict import frozendict

from b2s.lang import *

PYTHON_ENV = {"defaultdict": defaultdict}


def prelude_trace(msg: Value, val: Value, env: Env) -> Value:
    match msg:
        case VString(s):
            print(s)
        case _:
            raise NotImplementedError("TODO")
    return val


def prelude_foldl(func: Value, init: Value, foldable: Value, env: Env) -> Value:
    acc = init
    node = foldable

    while True:
        match node:
            case VNil():
                break
            case VCons(value, nxt):
                acc = apply_function_values(func, (acc, value), env)
                node = nxt
            case _:
                return VError("lst input to foldl is not a list")

    return acc


def prelude_head(lst: Value, env: Env) -> Value:
    match lst:
        case VNil():
            return VError("empty list does not have a head")
        case VCons(value, _):
            return value
        case _:
            return VError("lst input to head is not a list")


def prelude_log(v: Value, env: Env) -> VNumber | VError:
    match v:
        case VNumber(n):
            return VNumber(math.log(n))
        case _:
            return VError("input to log is not a number")


def prelude_exp(v: Value, env: Env) -> VNumber | VError:
    match v:
        case VNumber(n):
            return VNumber(math.exp(n))
        case _:
            return VError("input to exp is not a number")


def prelude_length(lst: Value, env: Env) -> VNumber | VError:
    match lst:
        case VNil():
            return VNumber(0)
        case VCons(_, nxt):
            next_len = prelude_length(nxt, env)
            if isinstance(next_len, VError):
                return next_len
            return VNumber(1 + next_len.value)
        case _:
            return VError("lst input to length is not a list")


def prelude_fst(pair: Value, env: Env) -> Value:
    match pair:
        case VPair((fst, _)):
            return fst
        case _:
            return VError("input to fst is not a pair")


def prelude_snd(pair: Value, env: Env) -> Value:
    match pair:
        case VPair((_, snd)):
            return snd
        case _:
            return VError("input to snd is not a pair")


def prelude_abs(v: Value, env: Env) -> Value:
    match v:
        case VNumber(n):
            return VNumber(abs(n))
        case _:
            return VError("input to abs is not a number")


def prelude_int(v: Value, env: Env) -> Value:
    match v:
        case VNumber(n):
            return VNumber(int(n))
        case _:
            return VError("input to int is not a number")


prelude: Env = frozendict(
    {
        "trace": VPreludeFunc(prelude_trace),
        "foldl": VPreludeFunc(prelude_foldl),
        "head": VPreludeFunc(prelude_head),
        "len": VPreludeFunc(prelude_length),
        "log": VPreludeFunc(prelude_log),
        "exp": VPreludeFunc(prelude_exp),
        "fst": VPreludeFunc(prelude_fst),
        "snd": VPreludeFunc(prelude_snd),
        "abs": VPreludeFunc(prelude_abs),
        "int": VPreludeFunc(prelude_int),
    }
)  # type: ignore


def eval_binop(binop: BinOpKinds, left: Value, right: Value) -> Value:
    def eval_ints(binop: BinOpKinds, left: int, right: int) -> Value:
        value: Value
        match binop:
            case BinOpKinds.ADD:
                value = VNumber(left + right)
            case BinOpKinds.SUB:
                value = VNumber(left - right)
            case BinOpKinds.MUL:
                value = VNumber(left * right)
            case BinOpKinds.DIV:
                value = (
                    VNumber(left / right) if right != 0 else VError("divide by zero")
                )
            case BinOpKinds.EQ:
                value = VBool(left == right)
            case BinOpKinds.NEQ:
                value = VBool(left != right)
            case BinOpKinds.LT:
                value = VBool(left < right)
            case BinOpKinds.GT:
                value = VBool(left > right)
            case BinOpKinds.LTE:
                value = VBool(left <= right)
            case BinOpKinds.GTE:
                value = VBool(left >= right)
            case BinOpKinds.POW:
                value = VNumber(left**right)
            case _:
                raise NotImplementedError("TODO")
        return value

    def eval_bools(binop: BinOpKinds, left: bool, right: bool) -> Value:
        value: Value
        match binop:
            case BinOpKinds.EQ:
                value = VBool(left == right)
            case BinOpKinds.NEQ:
                value = VBool(left != right)
            case BinOpKinds.AND:
                value = VBool(left and right)
            case BinOpKinds.OR:
                value = VBool(left or right)
            case _:
                raise NotImplementedError("TODO")
        return value

    match binop, left, right:
        case BinOpKinds.CONS, _, _:
            return VCons(left, right)
        case _, VNumber(l), VNumber(r):
            return eval_ints(binop, l, r)
        case _, VBool(l), VBool(r):
            return eval_bools(binop, l, r)
        case (_, VError(msg), _) | (_, _, VError(msg)):
            raise ValueError(msg)

    raise NotImplementedError("TODO")


def apply_function_values(func: Value, args: tuple[Value, ...], env: Env) -> Value:
    match func:
        case VClosure(params, body, new_env):
            # ok to directly set since it should always be overwritten on call
            assert len(params) == len(args), "wrong number of args"
            new_env = new_env | frozendict(zip(params, args))
            return eval_lang(body, new_env)
        case VPreludeFunc(f):
            return f(*args, env)
        case _:
            return VError(f"{func} is not a closure")


def apply_function(func: Expr, args: tuple[Expr, ...], env: Env) -> Value:
    return apply_function_values(
        eval_lang(func, env), tuple(map(lambda arg: eval_lang(arg, env), args)), env
    )


def create_closure(params: tuple[Id, ...], body: Expr, env: Env) -> Value:
    closure = VClosure(
        params, body, frozendict({k: v for k, v in env.items() if k not in params})
    )
    return closure


def let_binding(name: Id, expr: Expr, body: Expr, env: Env) -> Value:
    return eval_lang(body, frozendict({**env, name: eval_lang(expr, env)}))


def eval_if_else(cond: Expr, then: Expr, els: Expr, env: Env) -> Value:
    result = eval_lang(cond, env)
    match result:
        case VBool(truth_value):
            expr = then if truth_value else els
            return eval_lang(expr, env)
        case _:
            return VError("bool is not type of condition")


def memoize_expr(eval_fcn) -> Callable[[Expr, Env], Value]:
    class Wrapper:
        def __init__(self):
            self.memo = {}
            self.vals = {}

        def __call__(self, expr: Expr, env: Env) -> Value:
            args = (expr, env)
            if args not in self.memo:
                self.memo[args] = eval_fcn(expr, env)

                if isinstance(expr, ECall):
                    assert (
                        expr not in self.vals or self.vals[expr] == self.memo[args]
                    ), "call expr should always evaluate to same value"
                    self.vals[expr] = self.memo[args]

            if isinstance(self.memo[args], VError):
                assert False, f"error: {self.memo[args].msg}"
            return self.memo[args]

        def clear(self):
            self.memo.clear()
            self.vals.clear()

    return Wrapper()


def eval_lang(expr: Expr, env: Env) -> Value:
    match expr:
        case EVar(x) | EStream(x):
            return env.get(x, prelude.get(x, VError(f"{x} is not defined")))
        case ENumber(n):
            return VNumber(n)
        case EBool(b):
            return VBool(b)
        case EString(s):
            return VString(s)
        case EBinOp(left, op, right):
            return eval_binop(op, eval_lang(left, env), eval_lang(right, env))
        case ECall(func, args):
            return apply_function(func, args, env)
        case ELam(params, body):
            return create_closure(params, body, env)
        case ELet(name, expr, body):
            return let_binding(name, expr, body, env)
        case EIte(cond, then, els):
            return eval_if_else(cond, then, els, env)
        case ENil():
            return VNil()
        case EPair(elts):
            return VPair(tuple(eval_lang(elt, env) for elt in elts))
        case EMapUpdate(expr, mappings):
            # TODO: if `state` is a VError, then the error is lost in this assertion
            state = eval_lang(expr, env)
            assert isinstance(state, VMap), "state is not a map"

            state_dict = dict(state.value)
            for k, v in mappings.items():
                state_dict[k] = eval_lang(v, frozendict({**env, **state_dict}))
            return VMap(frozendict(state_dict))
        case EMapNil():
            return VMap(frozendict({}))
        case EMapGet(map_, key):
            # TODO: if `state` is a VError, then the error is lost in this assertion
            state = eval_lang(map_, env)
            assert isinstance(state, VMap), "state is not a map"
            assert key in state.value, f"{key} is not in map"
            return state.value.get(key, VError(f"{key} is not in map"))

        case EPythonExpr(ex):
            py_env = {k: to_python_value(v) for k, v in env.items()}
            val = eval(ex, {**PYTHON_ENV, **py_env["S"], **py_env})
            return to_value(val)

    raise NotImplementedError("TODO")
