import logging
import re
import time
from dataclasses import dataclass, replace
from itertools import groupby
from numbers import Number
from typing import Any, Generic, Literal, Self, Sequence, TypeVar

from frozendict import frozendict
from tqdm import tqdm

import b2s.ast_extra as ast
from b2s.const import VAR_NAME
from b2s.converters.converter import Converter
from b2s.converters.irs.inference import infer_relational_signature, sketchify
from b2s.expr_template import ConcreteExprGrammar
from b2s.input_config import InputConfig
from b2s.lang import ELam, VMap, fold_expr, pprint, to_python_value, to_value, to_value_list
from b2s.lang_interp import eval_lang
from b2s.pyast_to_lang import PyAstToIntermLang
from b2s.rfs import RelationalSignature, RelationalSignatureExpr
from b2s.sketch_types import (
    EnvState,
    IRUnknownSpec,
)
from b2s.stats import Stats
from b2s.synthesizers.enum_syn import synthesize_from_io
from b2s.synthesizers.qe_syn import IntermLangQEGrammarConstructor
from b2s.synthesizers.template_enum_syn import (
    synthesize_from_templates,
    synthesize_stun,
)
from b2s.utils import (
    PyAstSizeCounter,
    ast_size,
    find_function_by_name,
    generate_auction_object,
    generate_bid_object,
    generate_float_tuples,
    generate_person_object,
    generate_side_input_object,
    get_docstring,
    map_expr_to_partial_stream,
)

T = TypeVar("T")


def _generate_input(stream_type: str, len_stream: int):
    match stream_type:
        case "float":
            return generate_float_tuples(len_stream)
        case "(float, float)":
            return tuple(
                zip(
                    generate_float_tuples(len_stream), generate_float_tuples(len_stream)
                )
            )
        case "bid":
            return list(generate_bid_object(len_stream))
        case "auction":
            return list(generate_auction_object(len_stream))
        case "person":
            return list(generate_person_object(len_stream))
        case "side_input":
            return list(generate_side_input_object(len_stream))
            

    raise ValueError(f"Unknown stream type: {stream_type}")


@dataclass(slots=True, frozen=True)
class IncrementalExecState(Generic[T]):
    """
    EnvStates of two executions that differ by one input.

    :param fst: The pre-execution EnvState.
    :param snd: The post-execution EnvState.
    :param delta: The difference of the input.
    """

    fst: EnvState
    snd: EnvState
    delta: T


@dataclass(slots=True, frozen=True)
class ProgramInput(Generic[T]):
    stream: tuple[T, ...]
    auxiliary: frozendict[str, Any]

    def pop_stream(self) -> tuple[Self, T]:
        return replace(self, stream=self.stream[:-1]), self.stream[-1]

    @classmethod
    def create(cls, stream: Sequence[T], **aux):
        return cls(tuple(stream), frozendict(aux))


def float_number_is_good(n: float) -> bool:
    NUM_DIGITS = 4

    if not isinstance(n, Number):
        return True

    rounded = round(n, NUM_DIGITS)
    return rounded == n and repr(rounded) == repr(n)


class ObservationalFuncConverter(Converter):
    num_workers: int
    mode: Literal["partial", "complete"]

    def __init__(
        self,
        *,
        timeout: int,
        num_workers: int,
        syn_unroll_depth: int,
        syn_num_inputs: int,
        syn_ablation: Literal["none", "nodecomp", "nosymbolic"],
        mode: Literal["partial", "complete"],
        **_,
    ) -> None:
        self.num_workers = num_workers
        self.syn_unroll_depth = syn_unroll_depth
        self.syn_num_inputs = syn_num_inputs
        self.syn_ablation = syn_ablation
        self.mode = mode
        self.timeout = timeout

        assert self.syn_ablation in {"none", "nodecomp", "nosymbolic"}
        assert self.mode in {"partial", "complete"}

    def capture_exec_ios_alternative(
        self,
        prog: ELam,
        rel_sig,
        unk_specs: dict[str, IRUnknownSpec],
        num_input: int,
        input_config: InputConfig,
    ) -> dict[str, IRUnknownSpec]:
        OPTIONS_LEN_STREAM = [5, 6, 7, 8, 9, 10]
        OPTION_LEN_AUX_EXAMPLES = 15

        # expand aux args
        if len(input_config.args) == 1 and all(
            isinstance(t, str) and t.startswith("%")
            for t in input_config.args[0].values()
        ):
            for arg, t in list(input_config.args[0].items()):
                exs_t = _generate_input(t.removeprefix("%"), OPTION_LEN_AUX_EXAMPLES)
                input_config.args[0][arg] = exs_t

        progress_bar = tqdm(total=num_input * len(OPTIONS_LEN_STREAM))

        num_tried = 0
        num_accepted = 0
        inputs_sorted = set()
        inputs = []

        for len_stream in OPTIONS_LEN_STREAM:
            num_accepted = 0
            while num_accepted < num_input:
                currs = [
                    ProgramInput.create(
                        _generate_input(input_config.stream_type, len_stream),
                        **aux_args,
                    )
                    for _ in range(num_input)
                    for aux_args in input_config.args
                ]

                result = self.capture_exec_ios(prog, rel_sig, unk_specs, currs)

                for i, curr in enumerate(currs):
                    if num_accepted >= num_input:
                        break

                    for spec in result.values():
                        assert spec.io_examples is not None
                        ctx, v = spec.io_examples[i]
                        if not ctx and v is None:
                            break
                        if isinstance(v, complex) or any(
                            isinstance(v, complex) for v in ctx.values()
                        ):
                            # ignore complex numbers as they are not supported
                            break
                        if input_config.find_good_input and (
                            not float_number_is_good(v)
                            or any(not float_number_is_good(v) for v in ctx.values())
                        ):
                            break
                    else:
                        try:
                            xs = tuple(sorted(curr.stream))
                            if xs in inputs_sorted:
                                continue
                        except TypeError:
                            xs = tuple(curr.stream)

                        progress_bar.update()
                        num_accepted += 1
                        inputs.append(curr)
                        inputs_sorted.add(xs)

                num_tried += num_input
                progress_bar.set_description(f"tried {num_tried} inputs")

        return self.capture_exec_ios(prog, rel_sig, unk_specs, inputs)

    def capture_exec_ios(
        self,
        prog: ELam,
        rel_sig,
        unk_specs: dict[str, IRUnknownSpec],
        inputs: Sequence[ProgramInput],
    ) -> dict[str, IRUnknownSpec]:
        def _get_updated_vars(e) -> list[str]:
            return fold_expr(
                e,
                f_var=lambda v: [],
                f_stream=lambda v: [],
                f_int=lambda v: [],
                f_bool=lambda v: [],
                f_str=lambda v: [],
                f_binop=lambda left, op, right: left + right,
                f_call=lambda func, args: func + sum(args, []),
                f_lam=lambda params, body: body,
                f_let=lambda name, expr, body: expr + body,
                f_ite=lambda cond, then, els: cond + then + els,
                f_nil=lambda: [],
                f_pair=lambda elts: sum(elts, []),
                f_unk=lambda v: [],
                f_map=lambda _, exprs: sum(exprs.values(), list(exprs)),
                f_map_nil=lambda: [],
                f_map_get=lambda ex, _: ex,
                f_python_expr=lambda _: [],
            )

        def _extract_values(rse: RelationalSignatureExpr, value):
            py_v = to_python_value(value)
            match value:
                case VMap():
                    updated_vars = _get_updated_vars(rse.expand())
                    py_v = {k: v for k, v in py_v.items() if k in updated_vars}

                    return py_v[rse.map_idx] if rse.map_idx in py_v else py_v
            return py_v

        def _opt_extract_background_exprs(
            rse: RelationalSignatureExpr, value
        ) -> dict[str, Any]:
            """
            Speedup independent synthesis - extract background expressions that are
            independent of the unknown.

            Take variance as an example. sq_s can use s as a background expression.
            """
            py_v = to_python_value(value)
            match value:
                case VMap():
                    return {k: v for k, v in py_v.items() if k != rse.map_idx}
            return {}

        def _exec_input(
            prog,
            rel_sig: RelationalSignature,
            unk_specs: dict[str, IRUnknownSpec],
            input_,
        ):
            input_init, input_delta = input_.pop_stream()

            params_val = {
                k: _extract_values(
                    expr,
                    eval_lang(
                        expr.expand(),
                        frozendict(
                            {
                                **{
                                    k: to_value(v)
                                    for k, v in input_init.auxiliary.items()
                                },
                                VAR_NAME.INPUT_STREAM: to_value_list(input_init.stream),
                            }
                        ),
                    ),
                )
                for k, expr in rel_sig.items()
            }
            params_val[VAR_NAME.CURRENT_ELEMENT] = input_delta
            if isinstance(input_delta, tuple) and len(input_delta) == 2:
                params_val[VAR_NAME.CURRENT_ELEMENT_1] = input_delta[0]
                params_val[VAR_NAME.CURRENT_ELEMENT_2] = input_delta[1]
            params_val |= input_init.auxiliary

            unk_eval_result = {
                unk: eval_lang(
                    (
                        unk_spec.equivalent_expr.expand()
                        if self.mode == "complete"
                        else map_expr_to_partial_stream(
                            unk_spec.equivalent_expr
                        ).expand()
                    ),
                    frozendict(
                        {
                            **{k: to_value(v) for k, v in input_init.auxiliary.items()},
                            VAR_NAME.INPUT_STREAM: to_value_list(
                                [*input_init.stream, input_delta]
                            ),
                            "xs'": to_value_list(input_init.stream),
                        }
                    ),
                )
                for unk, unk_spec in unk_specs.items()
            }

            return [
                (
                    unk_id,
                    {
                        **_opt_extract_background_exprs(
                            unk_spec.equivalent_expr, unk_eval_result[unk_id]
                        ),
                        **params_val,
                    },
                    _extract_values(
                        unk_spec.equivalent_expr,
                        unk_eval_result[unk_id],
                    ),
                )
                for unk_id, unk_spec in unk_specs.items()
            ]

        examples = []
        for input_ in inputs:
            try:
                ps = _exec_input(prog, rel_sig, unk_specs, input_)
                examples.extend(ps)
            except ValueError as ex:
                logging.warning(f"Failed to execute input: {input_}\n{ex}")
                examples += [(unk_id, {}, None) for unk_id in unk_specs.keys()]
                continue
        examples = sorted(examples, key=lambda x: x[0])

        return {
            k: replace(unk_specs[k], io_examples=list(map(lambda x: x[1:], grp)))
            for k, grp in groupby(examples, lambda x: x[0])
        }

    def enum_solve(
        self: Self,
        unk_specs: dict[str, IRUnknownSpec],
        grammar: dict[str, ConcreteExprGrammar],
    ) -> dict[str, str]:
        """
        Solves the given synthesis specifications by STUN.

        Returns:
            A mapping from the name of unknowns to its synthesized program.
        """
        syn_results: dict[str, str] = {}
        for unk_name, spec in unk_specs.items():
            start_time = time.time()
            use_template = unk_name in grammar

            assert spec.io_examples is not None

            syn_result = (
                synthesize_from_templates(
                    spec.io_examples,
                    grammar[unk_name].exprs,
                )
                if use_template
                else synthesize_from_io(spec.io_examples, timeout=self.timeout)
            )
            if use_template and syn_result is None:
                syn_result = synthesize_stun(
                    spec.io_examples, grammar[unk_name].guards, grammar[unk_name].exprs
                )

            assert isinstance(syn_result, str) or syn_result is None or not syn_result
            if syn_result:
                logging.info(
                    f"Synthesized {unk_name} in {time.time() - start_time :.2f}s "
                    f"from {'template' if use_template else 'IO'}"
                )
                logging.info(syn_result)

                assert isinstance(syn_result, str)
                syn_results[unk_name] = syn_result
                continue
            logging.error(
                f"Failed to synthesize {unk_name} from {'template' if use_template else 'IO'}"
            )
        return syn_results

    def convert(self, py_src: str) -> Stats:
        """
        Converts an offline program to an online program with observational equivalence.
        """

        stats = Stats()

        parse_start_time = time.time()
        module = ast.parse(py_src)
        docstr = get_docstring(module)

        assert docstr is not None, "docstring is required"
        input_config = InputConfig.load_from_docstring(docstr)
        py_func = find_function_by_name(module, input_config.func)
        assert py_func is not None, f"cannot find function {input_config.func}"

        if input_config.unroll_depth is not None:
            self.syn_unroll_depth = input_config.unroll_depth
            logging.warning(f"unroll depth is set to {self.syn_unroll_depth}")

        func = PyAstToIntermLang.convert(
            py_src, input_config.func, input_config.stream_param
        )
        assert isinstance(func, ELam)
        rel_sig = infer_relational_signature(func)
        stats.offline_ast_size = ast_size(func)

        logging.info("*" * 62)
        logging.info("Inferred relational signature:")
        pad_width = max(map(len, rel_sig.keys()))
        for k, v in rel_sig.items():
            logging.info(f"\t{k.ljust(pad_width)} -> {pprint(v.expr)}")
        logging.info("*" * 62)

        sketch = sketchify(
            func, rel_sig, self.syn_ablation == "nodecomp", self.mode == "partial"
        )
        logging.info(f"Input: {pprint(func)}")
        logging.info(f"Sketch: {pprint(sketch.sketch)}")
        stats.parse_time = time.time() - parse_start_time
        stats.num_exprs_to_synthesize = len(sketch.unknowns)

        test_start_time = time.time()
        unk_specs = self.capture_exec_ios_alternative(
            func, rel_sig, sketch.unknowns, self.syn_num_inputs, input_config  # type: ignore
        )
        stats.test_time = time.time() - test_start_time

        qe_start_time = time.time()
        if self.syn_ablation == "nosymbolic":
            grammar = {}
        else:
            grammar_ctr = IntermLangQEGrammarConstructor(
                self.syn_unroll_depth, self.mode, input_config.stream_type
            )
            grammar = grammar_ctr.construct_grammar(func, rel_sig, unk_specs)
        stats.qe_time = time.time() - qe_start_time

        syn_start_time = time.time()
        syn_res = self.enum_solve(unk_specs, grammar)

        eq_classes_exprs = set()
        stats.online_ast_size = ast_size(sketch.sketch)

        logging.info("*" * 62)
        result = pprint(sketch.sketch)
        for unk_name, syn_prog in syn_res.items():
            result = re.sub(rf"\?\?{unk_name}\b", syn_prog, result)
            if self.mode == "complete":
                stats.exprs_sizes.append(PyAstSizeCounter().run(syn_prog))
            else:
                expr = sketch.unknowns[unk_name].equivalent_expr.expr
                if expr in eq_classes_exprs:
                    stats.exprs_sizes.append(0) # already counted
                    continue
                eq_classes_exprs.add(expr)
                partial_expr_size = ast_size(expr)
                stats.exprs_sizes.append(PyAstSizeCounter().run(syn_prog) + partial_expr_size)
            stats.online_ast_size += PyAstSizeCounter().run(syn_prog)

        stats.syn_time = time.time() - syn_start_time
        stats.total_time = time.time() - parse_start_time
        if len(syn_res) == len(unk_specs):
            stats.solution = result
        else:
            logging.error(f"partial solution: {result}")

        return stats
