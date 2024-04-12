cmake_minimum_required(VERSION 3.18)

# Set the project name
project(Evaluation)

# Read environment variables
set(TIMEOUT "$ENV{timeout}" CACHE STRING "Timeout in seconds")
set(INPUT_DIR "$ENV{input_dir}" CACHE STRING "Input directory path")
set(OUTPUT_DIR "$ENV{output_dir}" CACHE STRING "Output directory path")
set(TOOL_ARGS "$ENV{tool_args}" CACHE STRING "tool arguments")

if(NOT DEFINED ENV{SRC_EXT})
    message(FATAL_ERROR "You must set SRC_EXT environment variable")
endif()
set(SRC_EXT "$ENV{SRC_EXT}")

set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(benchmark)
file(GLOB input_files "${INPUT_DIR}/*.${SRC_EXT}")
foreach(input_file ${input_files})
  # Get the file name without the directory path
  get_filename_component(input_file_name ${input_file} NAME)
  get_filename_component(input_file_name_without_ext ${input_file_name} NAME_WE)

  # Define the command to run
  set(output_file "${OUTPUT_DIR}/${input_file_name_without_ext}.out")

  # Add the command to the build target
  add_custom_command(
    COMMAND ./timeout_runner ${input_file} ${output_file} ${TOOL_ARGS}
    DEPENDS ${input_file}
    OUTPUT ${output_file}
    WORKING_DIRECTORY ${ROOT_DIR}
  )

  add_custom_target(
    ${input_file_name}
    DEPENDS ${output_file}
  )

  add_dependencies(benchmark ${input_file_name})
endforeach()
