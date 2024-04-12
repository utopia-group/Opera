import ast
from typing import Any

from b2s.const import VAR_NAME
from b2s.utils import parse_term


class AstMutator(ast.NodeTransformer):
    pass


def compose_mutators(*mutators: AstMutator) -> AstMutator:
    """Compose multiple mutators into one."""

    class ComposedMutator(AstMutator):
        def visit(self, node):
            for mutator in mutators:
                node = mutator.visit(node)
            return node

    return ComposedMutator()


class MapInputAstMutator(AstMutator):
    def __init__(self) -> None:
        super().__init__()
        self.target_id = None

    def visit_For(self, node: ast.For) -> Any:
        match node:
            case ast.For(target=t, iter=i, body=b):
                match t, i:
                    case ast.Name(id=tid), ast.Name(id=VAR_NAME.INPUT_STREAM):
                        self.target_id = tid
                        for s in b:
                            self.generic_visit(s)
                        self.target_id = None
                        return node

        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id == self.target_id:
            return parse_term(f"({self.target_id} * 2 + 1)")

        return self.generic_visit(node)
