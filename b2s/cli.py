import logging
import random
from argparse import ArgumentParser, Namespace
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

from b2s.converters import (
    Converter,
    GptConverter,
    ObservationalConverter,
    ObservationalFuncConverter,
)

CONVERTERS = {
    "gpt": GptConverter,
    "observational": ObservationalConverter,
    "obs_func": ObservationalFuncConverter,
}


def parse_cli_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--timeout", type=int, default=timedelta(minutes=10).seconds)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--converter", type=str, default="obs_func")
    parser.add_argument("--output_path", "-o", type=Path, default=None)

    gpt_group = parser.add_argument_group("gpt")
    gpt_group.add_argument("--gpt_gen_sig", action="store_true", default=False)
    gpt_group.add_argument("--gpt_model", type=str, default="gpt-4")
    gpt_group.add_argument("--gpt_max_tokens", type=int, default=512)

    syn_group = parser.add_argument_group("synthesis")
    syn_group.add_argument("--syn_unroll_depth", type=int, default=3)
    syn_group.add_argument("--syn_num_inputs", type=int, default=20)
    syn_group.add_argument(
        "--syn_ablation",
        type=str,
        choices=["none", "nodecomp", "nosymbolic"],
        default="none",
    )

    parser.add_argument("src_path", type=Path)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    args = parse_cli_args()

    with args.src_path.open("r", encoding="utf8") as f:
        src = f.read()

    converter = CONVERTERS[args.converter](**args.__dict__, mode="complete")
    assert isinstance(converter, Converter)
    stats = converter.convert(src)

    if stats.solution is None and args.syn_ablation != "nodecomp":
        # can't apply the optimization with nodecomp
        converter = CONVERTERS[args.converter](**args.__dict__, mode="partial")
        stats_ = converter.convert(src)
        stats = stats.union(stats_)

    if args.output_path is not None:
        with args.output_path.open("w", encoding="utf8") as f:
            f.write(stats.json())

    logging.info(
        f"*************************** RESULT ***************************\n{stats.format()}"
    )


if __name__ == "__main__":
    random.seed(11)
    main()
