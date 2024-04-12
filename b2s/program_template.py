from typing import TypeVar, Generic
from dataclasses import dataclass
from itertools import chain

T = TypeVar("T")


@dataclass
class ProgramTemplate(Generic[T]):
    list_split: list[T]
    prev_init: list[T]
    prev_loop: list[T]
    prev_post: list[T]
    prev_rel_init: list[T]
    recover_loop_state: list[T]
    update_inductive_var: list[T]
    unrolled_loop: list[T]
    post: list[T]

    # auxiliary
    synthesis_vars: list[str]

    @classmethod
    def create(cls) -> "ProgramTemplate[T]":
        return cls(
            list_split=[],
            prev_init=[],
            prev_loop=[],
            prev_post=[],
            prev_rel_init=[],
            recover_loop_state=[],
            update_inductive_var=[],
            unrolled_loop=[],
            post=[],
            # auxiliary
            synthesis_vars=[],
        )

    def to_list(self) -> list[T]:
        return list(
            chain(
                self.list_split,
                self.prev_init,
                self.prev_loop,
                self.prev_post,
                self.prev_rel_init,
                self.recover_loop_state,
                self.update_inductive_var,
                self.unrolled_loop,
                self.post,
            )
        )

    def to_stream_version_list(self) -> list[T]:
        return list(
            chain(
                self.recover_loop_state,
                self.update_inductive_var,
                self.unrolled_loop,
                self.post,
            )
        )
