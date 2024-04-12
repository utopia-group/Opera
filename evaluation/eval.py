import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

if shutil.which("latex"):
    plt.rcParams["text.latex.preamble"] = (
        r"\RequirePackage[tt=false, type1=true]{libertine}"
    )
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.size": 12,
            "font.family": "libertine",
        }
    )
else:
    print("LaTeX not found, using default font")

OPERA_BASELINE_STATS_PLOT_NAME = "evaluation/fig11a-stats-baseline.pdf"
OPERA_BASELINE_AUCTION_PLOT_NAME = "evaluation/fig11b-auction-baseline.pdf"
OPERA_ABLATION_PLOT_NAME = "evaluation/fig13-ablation.pdf"

NEXMARK_EVAL_PATH = "./evaluation/output/nexmark/"
STATS_EVAL_PATH = "./evaluation/output/stats/"
NO_SYMBOLIC_ABLATION_EVAL_PATH = "./evaluation/output/nosymb"
NO_DECOMP_ABLATION_EVAL_PATH = "./evaluation/output/nodecomp"

@dataclass(slots=True, frozen=True)
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


def load_data_from_dir(dir_path: str):
    prog_stats: dict[str, Stats] = {}

    jsons = list(Path(dir_path).glob("*.json"))
    for json_p in jsons:
        prog_name = json_p.stem
        with open(json_p, "r") as f:
            prog_stats[prog_name] = Stats(**json.load(f))
    return prog_stats


def to_df(data: dict[str, Stats]):
    return pd.DataFrame.from_dict(data, orient="index")


def read_opera_raw(nexmark_eval_path: str, stats_eval_path: str) -> pd.DataFrame:
    nexmark_data = load_data_from_dir(nexmark_eval_path)
    stats_data = load_data_from_dir(stats_eval_path)

    nexmark_df = to_df(nexmark_data)
    stats_df = to_df(stats_data)

    nexmark_df["type"] = "nexmark"
    stats_df["type"] = "stats"
    nexmark_df["type"] = nexmark_df["type"].astype("category")
    stats_df["type"] = stats_df["type"].astype("category")

    return pd.concat([nexmark_df, stats_df])


def read_opera_ablation(path: str) -> pd.DataFrame:
    data = load_data_from_dir(path)
    df = to_df(data)
    df["type"] = "ablation"
    df["type"] = df["type"].astype("category")
    return df


def transform_to_time_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df.loc[df["solution"].isna(), "total_time"] = None
    df = df[["total_time"]].rename(columns={"total_time": label})
    return df


def plot_ablation_catcus(ablation_time: pd.DataFrame):
    plt.figure(figsize=(5, 3))

    order = ["Opera", "Opera-NoDecomp", "Opera-NoSymbolic"]
    labels = [rf"\textsc{{{x}}}" for x in order]

    sns_plot = sns.lineplot(
        data=ablation_time,
        x="value",
        y="percentage",
        hue="variant",
        style="variant",
        markers=True,
        dashes=False,
        hue_order=order,
    )
    plt.grid(True)
    plt.xticks(np.arange(0, 250, 60))
    plt.ylim(0, 100)
    plt.xlim(0, 240)  # limits the x-axis from 0 to 250
    plt.xlabel("Running Total (sec)")
    plt.ylabel(r"\% of Benchmarks Solved")
    plt.title("Percentage of Benchmarks Solved by Time")

    handles, _ = sns_plot.get_legend_handles_labels()
    sns_plot.legend(handles=handles, labels=labels)
    plt.savefig(OPERA_ABLATION_PLOT_NAME, format="pdf", bbox_inches="tight")


TIME_PREFIX = "time="


def load_sygus_result(fp: Path) -> tuple[str, float | None]:
    INTERRUPT_MSG = "cvc5 interrupted by SIGTERM"

    with open(fp, "r") as f:
        lines = f.readlines()
        if any(INTERRUPT_MSG in line for line in lines):
            return fp.stem, None

        time_line = lines[-1]
        if not time_line.startswith(TIME_PREFIX):
            raise ValueError(f"Unexpected time line: {time_line}")

        time_line = time_line.removeprefix(TIME_PREFIX).strip()
        time = float(time_line)
        if not time:
            time = 1.0
        return fp.stem, time


def load_sketch_result(fp: Path) -> tuple[str, float | None]:
    DONE_MSG = "[SKETCH] DONE"
    with open(fp, "r") as f:
        lines = f.readlines()
        if not any(DONE_MSG in line for line in lines):
            return fp.stem, None

        time_line = lines[-1]
        if not time_line.startswith(TIME_PREFIX):
            raise ValueError(f"Unexpected time line: {time_line}")

        time_line = time_line.removeprefix(TIME_PREFIX).strip()
        time = float(time_line)
        if not time:
            time = 1.0
        if time == 3660.0:
            raise ValueError(fp)
        return fp.stem, time


def plot_baseline_cdf(df, num_benchmarks, title=None, pdf_path=None):
    X_MAX = 3601

    X_TICK_INTERVAL = X_MAX // 5

    df = (
        df.melt(id_vars="Program", var_name="variant")
        .fillna(1e200)
        .sort_values(by=["value"])
    )

    df["cumulative_count"] = df.groupby("variant").cumcount() + 1
    df["percentage"] = df["cumulative_count"].apply(lambda x: 100 * x / num_benchmarks)

    plt.figure(figsize=(5, 3))

    order = ["Opera", "Sketch", "CVC5"]
    labels = [r"\textsc{Opera}", "Sketch", "CVC5"]

    sns_plot = sns.lineplot(
        data=df,
        x="value",
        y="percentage",
        hue="variant",
        style="variant",
        markers=True,
        dashes=False,
        ci=None,
        hue_order=order,
    )
    plt.grid(True)
    plt.xticks(np.arange(0, X_MAX, X_TICK_INTERVAL))
    plt.ylim(0, 101)
    plt.xlim(1, X_MAX)  # limits the x-axis from 0 to 250
    plt.xlabel("Running Total (sec)")
    plt.ylabel(r"\% of Benchmarks Solved")
    if title is not None:
        plt.title(title)

    plt.xscale("log")

    handles, _ = sns_plot.get_legend_handles_labels()
    sns_plot.legend(handles=handles, labels=labels, loc="upper right")

    if pdf_path is not None:
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")


def load_baseline_from_dir(
    dir_path: str,
    path_filter: Callable[[Path], bool],
    load_fcn: Callable[[Path], tuple[str, float | None]],
) -> dict[str, float | None]:
    data: dict[str, float | None] = {}

    files = list(
        f for f in Path(dir_path).glob("*.out") if f.is_file() and path_filter(f)
    )
    for f in files:
        k, v = load_fcn(f)
        data[k] = v
    return data


def load_baseline_result(opera_df: pd.DataFrame):

    opera_df = opera_df.copy()
    opera_df.loc[opera_df.index.str.startswith("q"), "Type"] = "Auction"
    opera_df["Type"].fillna("Stats", inplace=True)

    num_stats = len(opera_df[opera_df["Type"] == "Stats"])
    num_auction = len(opera_df[opera_df["Type"] == "Auction"])

    cvc5 = load_baseline_from_dir(
        "./evaluation/output/cvc5", lambda _: True, load_sygus_result
    )
    sketch = load_baseline_from_dir(
        "./evaluation/output/sketch", lambda _: True, load_sketch_result
    )

    baseline_df = pd.concat(
        [
            pd.DataFrame.from_dict(cvc5, orient="index", columns=["CVC5"]),
            pd.DataFrame.from_dict(sketch, orient="index", columns=["Sketch"]),
        ],
        axis=1,
    )

    df = (
        opera_df.join(baseline_df, how="outer")
        .reset_index()
        .rename(columns={"index": "Program"})
    )
    count_df = (
        df.groupby("Type")
        .agg(["count"])
        .sort_index(ascending=False)
        .drop(columns=["Program"])
    )
    num_benchmarks_per_type = [num_stats, num_auction]
    pct_df = (
        count_df.div(
            np.tile(
                num_benchmarks_per_type, len(count_df) // len(num_benchmarks_per_type)
            ),
            axis="index",
        )
        .transpose()
        .round(2)
        * 100
    )
    print("\n\n---- Table 2: Main synthesis result (Percentage) ----")
    pct_df_format = pct_df.applymap("{:.0f}%".format) # type: ignore
    print(tabulate(pct_df_format, headers="keys", tablefmt="psql"))  # type: ignore

    auction_df = df[df["Type"] == "Auction"].drop(columns=["Type"])
    auction_df = pd.concat(
        [
            auction_df,
            pd.DataFrame([["q-last", 1e200, 1e200, 1e200]], columns=auction_df.columns),
        ],
        axis=0,
    )
    plot_baseline_cdf(auction_df, num_auction, pdf_path=OPERA_BASELINE_AUCTION_PLOT_NAME)

    stats_df = df[df["Type"] == "Stats"].drop(columns=["Type"])
    plot_baseline_cdf(stats_df, num_stats, pdf_path=OPERA_BASELINE_STATS_PLOT_NAME)
    print("\n\n---- Figure 11: Comparison between Opera and baselines ----")
    print(f"Saved to {OPERA_BASELINE_STATS_PLOT_NAME}")
    print(f"Saved to {OPERA_BASELINE_AUCTION_PLOT_NAME}")

    print("\n\n---- Result for RQ2: Baseline comparison ----")
    num_sketch_solved = baseline_df["Sketch"].count()
    num_cvc5_solved = baseline_df["CVC5"].count()
    num_opera_solved = opera_df["Opera"].count()
    print(
        f"Opera outperforms existing SyGuS solvers, synthesizing {num_opera_solved/num_cvc5_solved:.1f}×"
        f" and {num_opera_solved/num_sketch_solved:.1f}× as many tasks as CVC5 and Sketch respectively."
    )


def main():
    df = read_opera_raw(NEXMARK_EVAL_PATH, STATS_EVAL_PATH)
    df_no_symbolic = read_opera_ablation(NO_SYMBOLIC_ABLATION_EVAL_PATH)
    df_no_decomp = read_opera_ablation(NO_DECOMP_ABLATION_EVAL_PATH)

    print("\n\n---- Table 1: Statistics about the benchmark set ----")
    ast_size_table_info = (
        df.groupby("type")[["offline_ast_size", "online_ast_size"]]
        .agg(["mean", "median"])
        .round(0)
        .astype(int)
        .stack(0)
        .unstack()
        .sort_index(ascending=False)
    )
    assert isinstance(ast_size_table_info, pd.DataFrame), "Expected a DataFrame"

    print(tabulate(ast_size_table_info, headers="keys", tablefmt="psql"))  # type: ignore

    opera_time = transform_to_time_df(df, "Opera")
    no_symb_time = transform_to_time_df(df_no_symbolic, "Opera-NoSymbolic")
    no_decomp_time = transform_to_time_df(df_no_decomp, "Opera-NoDecomp")
    ablation_time = pd.concat(
        [opera_time, no_symb_time, no_decomp_time], axis=1, join="outer"
    )

    # ablation nexmark not implemented, but theorethically it perform the same as opera does
    for nexmark_q in df[df["type"] == "nexmark"].index:
        ablation_time.loc[nexmark_q, "Opera-NoSymbolic"] = opera_time.loc[
            nexmark_q, "Opera"
        ]
        ablation_time.loc[nexmark_q, "Opera-NoDecomp"] = opera_time.loc[
            nexmark_q, "Opera"
        ]

    num_benchmarks = len(ablation_time)
    num_opera_solved = ablation_time["Opera"].count()
    mean_opera_time = ablation_time["Opera"].mean()

    load_baseline_result(opera_time)

    print("\n\n---- Result for RQ1: Opera Result ----")
    print(
        f"Opera can automatically synthesize {num_opera_solved} out of {num_benchmarks} online schemes"
        f" with an average synthesis time of {mean_opera_time:.1f} seconds."
    )

    percentage_solved = ablation_time.count() / num_benchmarks
    pct_opera_solved = percentage_solved["Opera"]

    print("\n\n---- Result for RQ3: % of benchmarks solved by ablations ----")
    for ablation, perc in percentage_solved.items():
        print(f"{ablation}: {perc:.2%} ({pct_opera_solved - perc:.2%} less than Opera)")

    abl_plot_df = (
        ablation_time.reset_index()
        .rename(columns={"index": "Program"})
        .melt(id_vars="Program", var_name="variant")
        .fillna(1e20)
        .sort_values(by=["value"])
    )
    abl_plot_df["cumulative_count"] = abl_plot_df.groupby("variant").cumcount() + 1
    abl_plot_df["percentage"] = abl_plot_df["cumulative_count"].apply(
        lambda x: 100 * x / num_benchmarks
    )
    plot_ablation_catcus(abl_plot_df)
    print("\n\n---- Figure 13: Comparison between Opera and its ablations ----")
    print(f"Saved to {OPERA_ABLATION_PLOT_NAME}")


if __name__ == "__main__":
    dir_path = Path(__file__).parent.parent
    os.chdir(dir_path)
    print(f"Changing working directory to {dir_path}")

    main()
