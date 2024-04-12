import ast

import pytest

from b2s.converters.ast_mutator import *


@pytest.mark.parametrize(
    "src, expected",
    [
        (
            r"""def mean(xs):
    s = 0
    for x in xs:
        s += x
    return s / len(xs)""",
            r"""def mean(xs):
    s = 0
    for x in xs:
        s += x * 2 + 1
    return s / len(xs)""",
        )
    ],
)
def test_map_input(src: str, expected: str) -> None:
    s = ast.parse(src)
    mutator = MapInputAstMutator()
    mutator.visit(s)
    assert ast.unparse(s) == expected
