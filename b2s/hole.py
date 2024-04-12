from dataclasses import dataclass


@dataclass(frozen=True)
class Hole:
    name: str
    terms: list[str]
