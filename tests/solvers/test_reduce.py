import pytest

from b2s.solvers.reduce import REDLOG_PARSER, run_reduce


@pytest.mark.parametrize("timeout", [5])
@pytest.mark.parametrize(
    "cmds,expected",
    [
        (["1+1;"], ["2"]),
        (["5**2;", "1+9;"], ["25", "10"]),
        (
            [
                r"eqn1 := prev_out * prev_len = (xs0 - prev_avg)^2 + (xs1 - prev_avg)^2;",
                r"eqn2 := sq_s = (xs0 - avg)^2 + (xs1 - avg)^2;",
                r"avg_defn := avg * (prev_len+1) = (xs0 + xs1 + x__) and prev_avg * prev_len = (xs0 + xs1);",
                r"phi := ex({xs0, xs1}, eqn1 and eqn2 and avg_defn);",
                r"rlqe phi;",
            ],
            [
                r"eqn1 := prev_len*prev_out = 2*prev_avg**2 - 2*prev_avg*xs0 - 2*prev_avg*xs1 "
                r"+ xs0**2 + xs1**2",
                r"eqn2 := sq_s = 2*avg**2 - 2*avg*xs0 - 2*avg*xs1 + xs0**2 + xs1**2",
                r"avg_defn := avg*prev_len + avg - x__ - xs0 - xs1 = 0 and prev_avg*prev_len - "
                r"xs0 - xs1 = 0",
                r"phi := ex(xs0,ex(xs1, - 2*prev_avg**2 + 2*prev_avg*xs0 + 2*prev_avg*xs1 + "
                r"prev_len*prev_out - xs0**2 - xs1**2 = 0 and  - 2*avg**2 + 2*avg*xs0 + "
                r"2*avg*xs1 + sq_s - xs0**2 - xs1**2 = 0 and (avg*prev_len + avg - x__ - xs0 - "
                r"xs1 = 0 and prev_avg*prev_len - xs0 - xs1 = 0)))",
                r"prev_avg**2*prev_len**2 - 4*prev_avg**2*prev_len + 4*prev_avg**2 - "
                r"2*prev_len* prev_out <= 0 and avg*prev_len + avg - prev_avg*prev_len - x__ = "
                r"0 and 2*avg**2 - 2*avg*prev_avg*prev_len + 2*prev_avg**2*prev_len - "
                r"2*prev_avg**2 + prev_len* prev_out - sq_s = 0",
            ],
        ),
    ],
)
def test_run_reduce(cmds: list[str], timeout: int, expected: list[str]) -> None:
    assert run_reduce(cmds, timeout) == expected


@pytest.mark.parametrize(
    "s, expected",
    [
        (
            (
                r"prev_avg**2*prev_len**2 - 4*prev_avg**2*prev_len + 4*prev_avg**2 - "
                r"2*prev_len* prev_out <= 0 and avg*prev_len + avg - prev_avg*prev_len - x__ = "
                r"0 and 2*avg**2 - 2*avg*prev_avg*prev_len + 2*prev_avg**2*prev_len - "
                r"2*prev_avg**2 + prev_len* prev_out - sq_s = 0 and "
                r"x + y <> 0",
                "((prev_avg) ** 2 * (prev_len) ** 2 - 4 * (prev_avg) ** 2 * prev_len + 4 * (prev_avg) ** 2 - "
                "2 * prev_len * prev_out <= 0 ∧ avg * prev_len + avg - prev_avg * prev_len - x__ = 0 ∧ 2 * (avg) ** 2"
                " - 2 * avg * prev_avg * prev_len + 2 * (prev_avg) ** 2 * prev_len - 2 * (prev_avg) ** 2 + "
                "prev_len * prev_out - sq_s = 0 ∧ x + y <> 0)",
            )
        ),
        (r"p = (s*( - s + 3))/3", r"p = ((s * (^-s + 3)) / 3)"),
    ],
)
def test_parser(s: str, expected: str) -> None:
    result = REDLOG_PARSER.parse(s)
    assert str(result) == expected
