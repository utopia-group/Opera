#!/usr/bin/env bash

source ./run_benchmark.sh

[[ -z "${NUM_PROC}" ]] && max_parallelism=$(nproc --all) || max_parallelism="${NUM_PROC}"

benchmark_name="benchmark_stats"
input_dir="../../benchmarks/stats"
stats_dir="../../evaluation/output/stats"
src_ext="py"
b2s_args=""

run_benchmark "${benchmark_name}" "${max_parallelism}" "${input_dir}" "${stats_dir}" "${src_ext}" "${b2s_args}"

benchmark_name="benchmark_nexmark"
max_parallelism=$num_processors
input_dir="../../benchmarks/nexmark"
stats_dir="../../evaluation/output/nexmark"
src_ext="py"
b2s_args=""

run_benchmark "${benchmark_name}" "${max_parallelism}" "${input_dir}" "${stats_dir}" "${src_ext}" "${b2s_args}"
