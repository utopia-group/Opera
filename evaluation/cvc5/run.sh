#!/usr/bin/env bash

set -e

export benchmark_name="evaltemp_cvc5"
export output_dir=$(realpath ../output/cvc5)
export input_dir=$(realpath ../../benchmarks/sygus/stats)
[[ -z "${NUM_PROC}" ]] && max_parallelism=$(nproc --all) || max_parallelism="${NUM_PROC}"
export tool_args=""
export SRC_EXT="sy"

export timeout="10m"
export proc_timeout="11m"

mkdir -p "${output_dir}" && mkdir -p "${input_dir}"
mkdir -p "${benchmark_name}" && cd "${benchmark_name}"

echo "Running benchmark ${benchmark_name}"
echo "parallelism ${max_parallelism}, input directory ${input_dir}, out directory ${output_dir}"
echo "tool_args ${tool_args}"
echo "source file extension ${SRC_EXT}"

# Run CMake to generate the Makefile
cmake .. || {
    # If there's an error, go back and remove the directory
    cd ..
    rm -rf "${benchmark_name}"
    exit 1
}

# Run make with the specified parallelism
make -j"${max_parallelism}" -k benchmark

# Return to the parent directory
cd .. && rm -rf "${benchmark_name}"
