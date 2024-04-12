import ast as _ast
import sys

# pyright: reportWildcardImportFromLibrary=false
from ast import *  # noqa: F403


class NamedMutableExpr(_ast.expr):
    if sys.version_info >= (3, 10):
        __match_args__ = (
            "name",
            "body",
        )
    _fields = ("name", "body")
    body: _ast.expr
    name: str


class Unparser(_ast._Unparser):  # type: ignore
    def visit_NamedMutableExpr(self, node: NamedMutableExpr):
        self.traverse(node.body)


def _unparse(ast_obj: _ast.AST) -> str:
    unparser = Unparser()
    return unparser.visit(ast_obj)


unparse = _unparse
