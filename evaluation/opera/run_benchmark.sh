function run_benchmark() {
    local benchmark_name=$1
    local max_parallelism=$2
    local input_dir=$3
    local stats_dir=$4
    local src_ext=$5
    local b2s_args=$6
    
    mkdir -p "${stats_dir}" && mkdir -p "${input_dir}"

    # Set environment variables
    export timeout="${timeout}"
    export input_dir=`cd "${input_dir}"; pwd`
    export stats_dir=`cd "${stats_dir}"; pwd`
    export b2s_args="${b2s_args}"
    export SRC_EXT="${src_ext}"

    # Create a directory for the benchmark
    mkdir -p "${benchmark_name}" && cd "${benchmark_name}"

    echo "Running benchmark ${benchmark_name}"
    echo "parallelism ${max_parallelism}, input directory ${input_dir}, statistics directory ${stats_dir}"
    echo "b2s arguments ${b2s_args}"
    echo "source file extension ${src_ext}"

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
}

