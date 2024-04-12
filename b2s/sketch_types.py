import ast
from dataclasses import dataclass
from typing import Any, Generic, NewType, Sequence, TypeAlias, TypeVar

from frozendict import frozendict

from b2s.lang import Expr
from b2s.rfs import RelationalSignatureExpr

FuncSketchType = TypeVar("FuncSketchType")
ExprType = TypeVar("ExprType", covariant=True)

Value: TypeAlias = Any
EnvState: TypeAlias = frozendict[str, Value]

IOExample: TypeAlias = tuple[EnvState, Value]
IOExamples: TypeAlias = Sequence[IOExample]




@dataclass(frozen=True, slots=True)
class UnknownSpec(Generic[ExprType]):
    """
    Specification of an unknown variable in a sketch.

    :param equivalent_expr: An expression that is equivalent to the unknown.
    :param io_examples: A list of input-output examples that the unknown must satisfy.
    """

    unk_id: str
    equivalent_expr: ExprType
    io_examples: IOExamples | None


@dataclass(frozen=True, slots=True)
class ImperativeUnknownSpec(UnknownSpec[ExprType]):
    """
    Specification of an unknown variable in a sketch.
    Hacky way to support imperative for-loops with multiple left-variable assignments;
    can only be used during synthesis.

    :param variable: The name of the left-value in a for-loop.
    """

    variable: str


@dataclass(frozen=True, slots=True)
class Sketch(Generic[FuncSketchType, ExprType]):
    sketch: FuncSketchType
    unknowns: dict[str, UnknownSpec[ExprType]]


@dataclass(frozen=True, slots=True)
class IRSketch(Sketch[Expr, RelationalSignatureExpr]):
    pass


@dataclass(frozen=True, slots=True)
class IRUnknownSpec(UnknownSpec[RelationalSignatureExpr]):
    pass


ASTSketch = Sketch[ast.FunctionDef, ast.expr | ast.stmt]
ASTUnknownSpec = UnknownSpec[ast.expr | ast.stmt]
ASTImperativeUnknownSpec = ImperativeUnknownSpec[ast.expr | ast.stmt]
