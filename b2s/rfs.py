from dataclasses import dataclass, field
from typing import NewType, Sequence

from b2s.lang import ELet, Expr

ImperativeRelationalSignature = NewType("ImperativeRelationalSignature", dict[str, str])


RSExprEnv = Sequence[tuple[str, Expr]]


@dataclass(frozen=True, slots=True, eq=True)
class RelationalSignatureExpr:
    expr: Expr
    env: RSExprEnv = field(default_factory=tuple)
    map_idx: str | None = None

    def expand(self) -> Expr:
        expr = self.expr
        for var_name, var_expr in reversed(self.env):
            expr = ELet(var_name, var_expr, expr)
        return expr


RelationalSignature = NewType("RelationalSignature", dict[str, RelationalSignatureExpr])
