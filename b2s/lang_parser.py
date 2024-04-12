import functools
from pathlib import Path
from typing import Dict, List, Tuple

from lark import Lark, Token, Transformer, Tree

from b2s.lang import *


def construct_lark_parser() -> Lark:
    with open(Path(__file__).parent.absolute() / "lang_parser.lark", "r") as f:
        return Lark(f, start="expr")


def get_value(node):
    if isinstance(node, Tree):
        node = node.children[0]
        if isinstance(node, Token):
            if node.type == "expr_list":
                return [get_value(x) for x in node.value]
            else:
                return node.value
    return node


class IRTransformer(Transformer):
    unknown_counter: int

    def __init__(self):
        super(Transformer).__init__()
        self.unknown_counter = 0

    def number(self, tree) -> Expr:
        return ENumber(float(tree[0].value))

    def string(self, tree) -> Expr:
        s = tree[0].value[1:-1]  # remove extra quotes on front/back
        return EString(s)

    def bool(self, tree) -> Expr:
        b = tree[0].value
        value = b == "true"
        return EBool(value)

    def var(self, tree) -> Expr:
        ident, = tree
        return EVar(ident)

    def binop_parser(self, op, tree) -> Expr:
        if len(tree) == 1:
            return tree
        else:
            expr1, expr2 = tree
            return EBinOp(expr1, BinOpKinds(op), expr2)

    pow = functools.partialmethod(binop_parser, "^")
    mul = functools.partialmethod(binop_parser, "*")
    div = functools.partialmethod(binop_parser, "/")
    add = functools.partialmethod(binop_parser, "+")
    sub = functools.partialmethod(binop_parser, "-")
    cons = functools.partialmethod(binop_parser, "::")
    eq = functools.partialmethod(binop_parser, "==")
    neq = functools.partialmethod(binop_parser, "!=")
    le = functools.partialmethod(binop_parser, "<")
    ge = functools.partialmethod(binop_parser, ">")
    leq = functools.partialmethod(binop_parser, "<=")
    geq = functools.partialmethod(binop_parser, ">=")
    _and = functools.partialmethod(binop_parser, "&&")
    _or = functools.partialmethod(binop_parser, "||")

    def call(self, tree) -> Expr:
        func, args = tree
        return ECall(func, args)

    def lam(self, tree) -> Expr:
        params, body = tree
        return ELam(params, body)

    def param_list(self, tree) -> Expr:
        return tree

    def let(self, tree) -> Expr:
        name, expr, body = tree
        return ELet(name, expr, body)

    def ite(self, tree) -> Expr:
        cond, then, els = tree
        return EIte(cond, then, els)

    def list(self, tree) -> Expr:
        values, = tree
        lst = ENil()
        for e in reversed(values):
            lst = EBinOp(e, BinOpKinds.CONS, lst)
        return lst

    def tuple(self, tree) -> Expr:
        elts, = tree
        return EPair(elts)  # type: ignore

    def unknown(self, tree) -> Expr:
        counter_value = self.unknown_counter
        self.unknown_counter += 1
        return EUnknown(str(counter_value))

    def map_update(self, tree) -> Expr:
        expr, *mappings = tree
        mappings_dict = frozendict(mappings)
        return EMapUpdate(expr, mappings_dict)

    def map(self, tree) -> Expr:
        kvs = tree
        if kvs == [None]:
            return EMapNil()
        else:
            return EMapUpdate(EMapNil(), kvs)

    def map_get(self, tree) -> Expr:
        map_, key = tree
        return EMapGet(map_, key)

    # Helpers
    def ident(self, tree) -> Id:
        return Id(tree[0].value)

    def expr_list(self, tree) -> List[Expr]:
        if tree == [None]:
            return []
        return tree

    def key_value_list(self, tree) -> Dict[str, Expr]:
        if tree == [None]:
            return {}
        return dict(tree)

    def key_value(self, tree) -> Tuple[str, Expr]:
        key, value = tree
        return (key, value)


def parse_lark_repr(tree: Tree) -> Expr:
    transformer = IRTransformer()
    result_tree = transformer.transform(tree)
    return result_tree


def parse(text: str) -> Expr:
    ir_parser = construct_lark_parser()
    return parse_lark_repr(ir_parser.parse(text))
