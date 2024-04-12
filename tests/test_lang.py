import pytest
from frozendict import frozendict

from b2s.lang import to_value
from b2s.lang_interp import eval_lang
from b2s.lang_parser import parse


@pytest.mark.parametrize(
    "prog,expected,env",
    [
        ("1.7", 1.7, None),
        ("1", 1, None),
        ('"hello"', "hello", None),
        ("true", True, None),
        ("false", False, None),
        (r"(\x y -> (x + y))(x, y)", 3, {"x": 1, "y": 2}),
        (r"(\x y -> (x + y))(1, 4)", 5, None),
        (
            r'let f = \xs -> let S = {} in let S = foldl(\S n -> S{mn = if (n < mn) then n else mn}, S{mn = 1000}, xs) in S[mn] in f(xs)',
            8,
            {"xs": [90, 80, 8]},
        ),
        (
            r'let mean = \xs -> let S = {} in let S = foldl(\S n -> S{s = (s + n)}, S{s = 0}, xs) in (S[s] / len(xs)) in mean(xs)',
            2.0,
            {"xs": [1.0, 2.0, 3.0]},
        ),
        (
            r"""
let f = 
    (\xs -> let S = {} in
         let S = foldl(\S x -> S{s = (s + x)}, S{s = 0}, xs) in
            let S = foldl(\S x -> S{sq_s = (sq_s + ((x - avg) ^ 2))}, S{avg = (s / len(xs)); sq_s = 0}, xs) in
                (S[sq_s] / len(xs))) in f(xs)""",
            0.5,
            {"xs": [2.0, 3.0, 3.0, 4.0]},
        ),
    ],
)
def test_lang(prog: str, expected: str, env: dict | None):
    if env is not None:
        env = {k: to_value(v) for k, v in env.items()}
    assert eval_lang(parse(prog), frozendict(env or {})) == to_value(expected)