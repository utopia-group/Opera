import json
from dataclasses import asdict, dataclass, field, replace
from typing import Self


@dataclass(slots=True)
class Stats:
    solution: str | None = None
    qe_time: float = 0.0
    syn_time: float = 0.0
    parse_time: float = 0.0
    test_time: float = 0.0
    total_time: float = 0.0
    num_exprs_to_synthesize: int = 0
    exprs_sizes: list[int] = field(default_factory=list)
    offline_ast_size: int = 0
    online_ast_size: int = 0

    def union(self, other: Self) -> Self:
        if self.solution is None:
            return replace(
                other,
                qe_time=self.qe_time + other.qe_time,
                syn_time=self.syn_time + other.syn_time,
                parse_time=self.parse_time + other.parse_time,
                test_time=self.test_time + other.test_time,
                total_time=self.total_time + other.total_time,
            )

        return self

    def format(self: Self) -> str:
        return "\n".join(
            (
                "*" * 30,
                f"  Solution: {self.solution}",
                "*" * 30,
                f"   QE time: {self.qe_time:.3f}s",
                f"  Syn time: {self.syn_time:.3f}s",
                f"Parse time: {self.parse_time:.3f}s",
                f" Test time: {self.test_time:.3f}s",
                f"Total time: {self.total_time:.3f}s",
                "*" * 30,
                f"   # exprs: {self.num_exprs_to_synthesize}",
                f" expr size: {self.exprs_sizes}",
                f"  input sz: {self.offline_ast_size}",
                f" output sz: {self.online_ast_size}",

            )
        )

    def json(self) -> str:
        return json.dumps(asdict(self), indent=2)
