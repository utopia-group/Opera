# https://github.com/MaigoAkisame/enumerate-expressions

import random
import signal
from dataclasses import dataclass
from enum import Flag, auto
from fractions import Fraction


@dataclass(frozen=False, slots=True)
class BoxedString:
    value: str

    def __str__(self):
        return self.value

    def __repr__(self):
        return repr(self.value)


class Operator(Flag):
    PLUS = auto()
    MINUS = auto()
    TIMES = auto()
    DIVIDE = auto()
    LOG = auto()
    SQRT = auto()
    EXP = auto()
    POW = auto()

    def __str__(self):
        match self:
            case Operator.PLUS:
                return "+"
            case Operator.MINUS:
                return "-"
            case Operator.TIMES:
                return "*"
            case Operator.DIVIDE:
                return "/"
            case Operator.LOG:
                return "log"
            case Operator.EXP:
                return "exp"
            case Operator.POW:
                return "**"
            case Operator.SQRT:
                return "sqrt"

    def __contains__(self, other) -> bool:
        if other is None:
            return False
        return super().__contains__(other)


OPERATOR_MAP = {
    Operator.PLUS: "+",
    Operator.MINUS: "-",
    Operator.TIMES: "*",
    Operator.DIVIDE: "/",
    Operator.LOG: "log",
    Operator.EXP: "exp",
    Operator.POW: "**",
    Operator.SQRT: "sqrt",
}
REVERSED_OPERATOR_MAP = {v: k for k, v in OPERATOR_MAP.items()}


COMPONENTS = (
    Operator.PLUS
    | Operator.MINUS
    | Operator.TIMES
    | Operator.DIVIDE
    | Operator.LOG
    | Operator.EXP
    | Operator.POW
)


def _set_components(components: list[str]):
    global COMPONENTS
    COMPONENTS = None  # type: ignore
    for c in components:
        if COMPONENTS is None:
            COMPONENTS = REVERSED_OPERATOR_MAP[c]
        else:
            COMPONENTS |= REVERSED_OPERATOR_MAP[c]


# COMPONENTS = (
#     Operator.PLUS
#     | Operator.MINUS
#     | Operator.TIMES
#     | Operator.DIVIDE
#     | Operator.LOG
#     | Operator.EXP
# )
# COMPONENTS = Operator.PLUS | Operator.MINUS | Operator.TIMES | Operator.DIVIDE
# COMPONENTS = Operator.POW


TIMES_DIVIDE = Operator.TIMES | Operator.DIVIDE
PLUS_MINUS = Operator.PLUS | Operator.MINUS
PLUS_MINUS_TIMES_DIVIDE = PLUS_MINUS | TIMES_DIVIDE
TIMES_MINUS = Operator.TIMES | Operator.MINUS


class timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def eval_with_timeout(expr, global_vars=None, seconds=0.1):
    try:
        with timeout(seconds):
            return eval(expr, global_vars.copy() if global_vars else None)
    except TimeoutError:
        return "timeout"
    except:
        return "error"


class Node:
    var: BoxedString

    def __init__(self, op=None, left=None, right=None, polar=0, id=0, var=None):
        self.op = op
        self.var = var
        self.left = left
        self.right = right
        self.polar = polar
        self.id = id

    def __str__(self):
        if self.op is None:
            return str(self.var)
        if self.op in {Operator.LOG, Operator.EXP}:
            return str(self.op) + "(" + str(self.left) + ")"
        if self.op == Operator.POW:
            return f"(({self.left}) ** ({self.right}))"
        left = str(self.left)
        right = str(self.right)
        if self.op in TIMES_DIVIDE and self.left.op in PLUS_MINUS:
            left = "(" + left + ")"
        if (
            self.op == Operator.DIVIDE
            and self.right.op in PLUS_MINUS_TIMES_DIVIDE
            or self.op in TIMES_MINUS
            and self.right.op in PLUS_MINUS
        ):
            right = "(" + right + ")"
        return left + " " + str(self.op) + " " + right


def smart2(left, right):
    for op in {Operator.PLUS, Operator.TIMES}:
        if op != left.op and (op != right.op or left.id < right.left.id):
            yield Node(op, left, right)


def smart4(left, right):
    if (
        not left.op in PLUS_MINUS
        and right.op != Operator.MINUS
        and (right.op != Operator.PLUS or left.id < right.left.id)
    ) and Operator.PLUS in COMPONENTS:
        if left.polar == 0 or right.polar == 0:
            yield Node(Operator.PLUS, left, right, left.polar + right.polar)
        else:
            yield Node(Operator.PLUS, left, right, right.polar)

    if (
        left.op != Operator.MINUS and right.op != Operator.MINUS
    ) and Operator.MINUS in COMPONENTS:
        if left.polar == 0 and right.polar == 0:
            yield Node(Operator.MINUS, left, right, 1)
            yield Node(Operator.MINUS, right, left, -1)
        else:
            if left.polar == 0:
                yield Node(Operator.MINUS, right, left, right.polar)

            if right.polar == 0:
                yield Node(Operator.MINUS, left, right, left.polar)

    if (
        not left.op in TIMES_DIVIDE
        and right.op != Operator.DIVIDE
        and (right.op != Operator.TIMES or left.id < right.left.id)
    ) and Operator.TIMES in COMPONENTS:
        if left.polar == 0 or right.polar == 0:
            yield Node(Operator.TIMES, left, right, left.polar + right.polar)

        elif left.polar > 0:
            yield Node(Operator.TIMES, left, right, right.polar)

    if (
        left.op != Operator.DIVIDE and right.op != Operator.DIVIDE
    ) and Operator.DIVIDE in COMPONENTS:
        if left.polar == 0 or right.polar == 0:
            yield Node(Operator.DIVIDE, left, right, left.polar + right.polar)

            yield Node(Operator.DIVIDE, right, left, left.polar + right.polar)
        else:
            if left.polar > 0:
                yield Node(Operator.DIVIDE, left, right, right.polar)
            if right.polar > 0:
                yield Node(Operator.DIVIDE, right, left, left.polar)

    if Operator.EXP in COMPONENTS:
        yield Node(Operator.EXP, left, None)
        yield Node(Operator.EXP, right, None)
    if Operator.LOG in COMPONENTS:
        yield Node(Operator.LOG, left, None)
        yield Node(Operator.LOG, right, None)
    if Operator.POW in COMPONENTS:
        yield Node(Operator.POW, left, right)
        yield Node(Operator.POW, right, left)
    if Operator.SQRT in COMPONENTS:
        yield Node(Operator.SQRT, left, None)
        yield Node(Operator.SQRT, right, None)


def enum(actions, variables: list[BoxedString]):
    def DFS(trees, minj):
        if len(trees) == 1:
            yield trees[0]
            return
        for j in range(minj, len(trees)):
            for i in range(j):
                for node in actions(trees[i], trees[j]):
                    node.id = trees[-1].id + 1
                    new_trees = [
                        treesk for k, treesk in enumerate(trees) if k != i and k != j
                    ] + [node]

                    new_minj = j - 1 if actions in (smart2, smart4) else 1

                    for expression in DFS(new_trees, new_minj):
                        yield expression

    trees = [Node(var=v, id=i) for i, v in enumerate(variables)]
    return DFS(trees, 1)


def enum_with_dedup(actions, variables: list[BoxedString]):
    vals = set()
    random_ints = [random.randint(0, 10) for _ in range(len(variables))]
    env = {}
    for i, v in enumerate(variables):
        env[str(v)] = random_ints[i]
    exec("from math import log, exp", env)

    for expr in enum(actions, variables):
        val = eval_with_timeout(str(expr), env, 0.01)
        if val not in vals or val != "timeout":
            vals.add(val)
            yield expr


def main(n):
    """
    >>> main(4)
    Expressions with +, -, * and /:
      Smart: 1170 expressions, 1170 distinct expressions, 1170 distinct values
    """

    global a, b, c, d, e, f, g, h
    a = Fraction(3141592)
    b = Fraction(6535897)
    c = Fraction(9323846)
    d = Fraction(2643383)
    e = Fraction(2795028)
    f = Fraction(8419716)
    g = Fraction(9399375)
    h = Fraction(1058209)

    variables = [chr(ord("a") + i) for i in range(n)]

    print("Expressions with +, -, * and /:")

    smart_exps = list(map(str, enum_with_dedup(smart4, variables)))
    print(smart_exps)
    smart_uniq_exps = set(smart_exps)
    smart_uniq_values = set(
        eval_with_timeout(ex, seconds=0.2) for ex in smart_uniq_exps
    )
    print(
        f"  Smart: {len(smart_exps)} expressions, {len(smart_uniq_exps)} distinct expressions, {len(smart_uniq_values)} distinct values"
    )


if __name__ == "__main__":
    from doctest import testmod

    # testmod()

    n = int(input("Input the number of variables: "))

    main(n)
