cmake_minimum_required(VERSION 3.24)
project(flashvad_native C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT APPLE)
  message(FATAL_ERROR "The Accelerate runtime is available only on Apple platforms")
endif()

set(FLASHVAD_WEIGHTS_DIR "" CACHE PATH "Directory containing generated weights")
if(NOT EXISTS "${FLASHVAD_WEIGHTS_DIR}/flashvad_weights.c")
  message(FATAL_ERROR "Set FLASHVAD_WEIGHTS_DIR to an export-native output directory")
endif()

add_library(flashvad_native STATIC
  flashvad_native.c
  "${FLASHVAD_WEIGHTS_DIR}/flashvad_weights.c"
)
target_include_directories(flashvad_native PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${FLASHVAD_WEIGHTS_DIR}"
)
target_compile_options(flashvad_native PRIVATE -O3)
target_compile_definitions(flashvad_native PRIVATE ACCELERATE_NEW_LAPACK)
target_link_libraries(flashvad_native PUBLIC "-framework Accelerate")

add_executable(flashvad_benchmark benchmark.c)
target_link_libraries(flashvad_benchmark PRIVATE flashvad_native)
