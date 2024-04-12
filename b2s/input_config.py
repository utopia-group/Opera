import tomllib
from typing import Any, NewType

from attrs import define, field

from b2s.const import VAR_NAME
from b2s.expr_template import ExprTemplate
from b2s.rfs import ImperativeRelationalSignature
from b2s.synthesizers.smart_expr_enumerator import _set_components

SynthesisFacts = NewType("SynthesisFacts", dict[str, str])
LoopVariableMap = NewType("LoopVariableMap", dict[str, str])


@define(frozen=True)
class InputConfig:
    """Ground truth for an input program."""

    func: str
    stream_param: str = field(default=VAR_NAME.INPUT_STREAM)
    args: list[dict[str, Any]] = field(factory=lambda: [{}])
    relational_sig: ImperativeRelationalSignature = field(
        factory=lambda: ImperativeRelationalSignature({})
    )
    stream_type: str = "float"
    find_good_input: bool = False
    synthesis_facts: SynthesisFacts | None = None
    expr_templates: dict[str, list[ExprTemplate]] | None = None
    loop_var_map: LoopVariableMap | None = None
    unroll_depth: int | None = None

    @classmethod
    def load_from_docstring(cls, docstring: str) -> "InputConfig":
        data = tomllib.loads(docstring)

        if "operators" in data:
            _set_components(data["operators"])
            del data["operators"]

        expr_templates = data.setdefault("expr_templates", {})
        for k in expr_templates:
            expr_templates[k] = [
                ExprTemplate.from_expr_str(**t) for t in expr_templates[k]
            ]

        return cls(**data)
