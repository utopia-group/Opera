#!/usr/bin/env bash

source ./run_benchmark.sh

[[ -z "${NUM_PROC}" ]] && max_parallelism=$(nproc --all) || max_parallelism="${NUM_PROC}"

benchmark_name="benchmark_stats_nodecomp"
input_dir="../../benchmarks/stats"
stats_dir="../../evaluation/output/nodecomp"
src_ext="py"
b2s_args="--syn_ablation=nodecomp"

run_benchmark "${benchmark_name}" "${max_parallelism}" "${input_dir}" "${stats_dir}" "${src_ext}" "${b2s_args}"
