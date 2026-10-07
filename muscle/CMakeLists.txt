# MUSCLE5 GPU -- T00 build freeze + GPU backend build entry.
#
# Design rules (see HANDOFF_AI.md sections 2, 6 and planning/tasks/T00_baseline_build.md):
#   * The compile-unit list is explicit (cmake/sources.cmake, extracted from the
#     R0 muscle.vcxproj).  No file(GLOB) over src/ -- stale tests and duplicate
#     main() units must never be pulled in.
#   * CPU-only configuration must not require any CUDA toolchain.
#   * OpenMP is a hard requirement (the upstream code includes <omp.h> from
#     myutils.h); no empty OpenMP stub is ever substituted.
#   * gitver.txt is generated in the build tree only; src/ is never written to,
#     and no upstream commit hash is invented.
#   * Default flags keep upstream semantics: no -ffast-math.  MUSCLE_STRICT_FP
#     adds -ffp-contract=off for a separately named strict reference build.
#   * MUSCLE_CUDA=ON adds -fmad=false --ftz=false --prec-div=true --prec-sqrt=true
#     for the CUDA translation units (no fast math, no TF32/FP16).

cmake_minimum_required(VERSION 3.18)

project(muscle5_gpu LANGUAGES CXX)

include(cmake/sources.cmake)

if(NOT MUSCLE_CORE_UNITS)
  message(FATAL_ERROR "cmake/sources.cmake did not define MUSCLE_CORE_UNITS")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MUSCLE_CUDA "Build the CUDA pair backend (requires nvcc + a CUDA device)" OFF)
option(MUSCLE_SANITIZE "Build with AddressSanitizer+UndefinedBehaviorSanitizer" OFF)
option(MUSCLE_STRICT_FP "Add -ffp-contract=off (strict FP reference build, separate identity)" OFF)
option(MUSCLE_ENABLE_WARNINGS "Enable -Wall -Wextra for the new T00+ code only" OFF)
set(MUSCLE_GITVER "archive-R0"
    CACHE STRING "Value written to build/generated/gitver.txt (provenance of this tree)")

# ---------------------------------------------------------------------------
# Compiler identification / recorded flags
# ---------------------------------------------------------------------------

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  set(MUSCLE_GNU_LIKE ON)
else()
  set(MUSCLE_GNU_LIKE OFF)
endif()

set(MUSCLE_BASE_FLAGS "")
if(MUSCLE_GNU_LIKE)
  list(APPEND MUSCLE_BASE_FLAGS -fno-strict-aliasing)
  if(MUSCLE_STRICT_FP)
    list(APPEND MUSCLE_BASE_FLAGS -ffp-contract=off)
  endif()
endif()

set(MUSCLE_SAN_FLAGS "")
if(MUSCLE_SANITIZE)
  if(NOT MUSCLE_GNU_LIKE)
    message(FATAL_ERROR "MUSCLE_SANITIZE requires GCC or Clang")
  endif()
  list(APPEND MUSCLE_SAN_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer -g)
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

# ---------------------------------------------------------------------------
# Generated provenance header (build tree only)
# ---------------------------------------------------------------------------

set(MUSCLE_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${MUSCLE_GENERATED_DIR}")
file(WRITE "${MUSCLE_GENERATED_DIR}/gitver.txt" "\"${MUSCLE_GITVER}\"\n")
message(STATUS "MUSCLE_GITVER = ${MUSCLE_GITVER} (generated, not an upstream commit)")

# ---------------------------------------------------------------------------
# Core library (224 R0 compile units, no main)
# ---------------------------------------------------------------------------

set(MUSCLE_CORE_SOURCES "")
foreach(unit ${MUSCLE_CORE_UNITS})
  list(APPEND MUSCLE_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/${unit}")
endforeach()

# GPU backend: pure C++ control layer, always compiled (it must also build with
# MUSCLE_CUDA=OFF so that `-backend cpu` and the CPU oracle work everywhere).
set(MUSCLE_GPU_CPP_SOURCES
  src/gpu/sha256.cpp
  src/gpu/stage_trace.cpp
  src/gpu/hmm_snapshot.cpp
  src/gpu/pair_reference.cpp
  src/gpu/posterior_finalize.cpp
  src/gpu/pair_backend_cpu.cpp
  src/gpu/batch_scheduler.cpp
  src/gpu/backend_registry.cpp
  src/gpu/dispatch.cpp
  src/gpu/super5_adapter.cpp
)
foreach(src ${MUSCLE_GPU_CPP_SOURCES})
  list(APPEND MUSCLE_CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
endforeach()

if(NOT MUSCLE_CUDA)
  list(APPEND MUSCLE_CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/pair_backend_cuda_stub.cpp")
endif()

add_library(muscle_core STATIC ${MUSCLE_CORE_SOURCES})

target_include_directories(muscle_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${MUSCLE_GENERATED_DIR}"
)

target_link_libraries(muscle_core PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(muscle_core PRIVATE ${MUSCLE_BASE_FLAGS} ${MUSCLE_SAN_FLAGS})
target_link_options(muscle_core PUBLIC ${MUSCLE_SAN_FLAGS})

# The new T00+ code is warning-clean under -Wall -Wextra; the R0 sources are not
# (they are frozen, so warnings there are expected and deliberately not tuned).
if(MUSCLE_ENABLE_WARNINGS)
  set_source_files_properties(${MUSCLE_GPU_CPP_SOURCES} PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra")
endif()

# ---------------------------------------------------------------------------
# Programs
# ---------------------------------------------------------------------------

add_executable(muscle "${CMAKE_CURRENT_SOURCE_DIR}/src/${MUSCLE_MAIN_UNITS}")
target_link_libraries(muscle PRIVATE muscle_core)

# Modified 2026-10-08: drop-in CLI frontend with a CUDA-first default.
add_executable(muscle_gpu "${CMAKE_CURRENT_SOURCE_DIR}/src/${MUSCLE_MAIN_UNITS}")
target_compile_definitions(muscle_gpu PRIVATE MUSCLE_GPU_FRONTEND=1)
target_link_libraries(muscle_gpu PRIVATE muscle_core)
install(TARGETS muscle muscle_gpu RUNTIME DESTINATION bin)

add_executable(omp_probe "${CMAKE_CURRENT_SOURCE_DIR}/tests/omp_probe.cpp")
target_link_libraries(omp_probe PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(omp_probe PRIVATE ${MUSCLE_BASE_FLAGS} ${MUSCLE_SAN_FLAGS})
target_link_options(omp_probe PRIVATE ${MUSCLE_SAN_FLAGS})

add_executable(pair_oracle
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/pair_oracle.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/dump_compare.cpp")
target_link_libraries(pair_oracle PRIVATE muscle_core)

add_executable(backend_contract "${CMAKE_CURRENT_SOURCE_DIR}/tests/backend_contract.cpp")
target_link_libraries(backend_contract PRIVATE muscle_core)

if(MUSCLE_ENABLE_WARNINGS)
  foreach(t omp_probe pair_oracle backend_contract)
    target_compile_options(${t} PRIVATE -Wall -Wextra)
  endforeach()
endif()

# ---------------------------------------------------------------------------
# CUDA backend (T04/T05)
# ---------------------------------------------------------------------------

if(MUSCLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "MUSCLE_CUDA=ON but no CUDA compiler was found. "
      "Install the CUDA toolkit or configure with -DMUSCLE_CUDA=OFF.")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()

  set(MUSCLE_CUDA_SOURCES
    src/gpu/cuda_context.cu
    src/gpu/pair_forward.cu
    src/gpu/pair_backward.cu
    src/gpu/pair_backend_cuda.cu
  )
  add_library(muscle_gpu_cuda STATIC ${MUSCLE_CUDA_SOURCES})
  target_include_directories(muscle_gpu_cuda PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${MUSCLE_GENERATED_DIR}"
  )
  target_link_libraries(muscle_gpu_cuda PUBLIC muscle_core)
  set_target_properties(muscle_gpu_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    POSITION_INDEPENDENT_CODE ON
  )
  # Strict FP32: --use_fast_math is never passed (nvcc's flag takes no argument,
  # and fast math is off by default), no FMA contraction, no FTZ, precise
  # div/sqrt.  These are recorded in muscle_build_config.txt below.
  set(MUSCLE_CUDA_STRICT_FLAGS
      --fmad=false --ftz=false --prec-div=true --prec-sqrt=true)
  target_compile_options(muscle_gpu_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${MUSCLE_CUDA_STRICT_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fno-strict-aliasing>
  )
  target_link_libraries(muscle PRIVATE muscle_gpu_cuda)
  target_link_libraries(muscle_gpu PRIVATE muscle_gpu_cuda)
  target_link_libraries(pair_oracle PRIVATE muscle_gpu_cuda)
  target_link_libraries(backend_contract PRIVATE muscle_gpu_cuda)
  target_compile_definitions(muscle_core PUBLIC MUSCLE_CUDA_ENABLED=1)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

enable_testing()

find_package(Python3 COMPONENTS Interpreter QUIET)

add_test(NAME vcxproj_source_list
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/gen_sources_from_vcxproj.py" --check)
set_tests_properties(vcxproj_source_list PROPERTIES LABELS "static")

add_test(NAME omp_probe_2threads COMMAND omp_probe 2)
set_tests_properties(omp_probe_2threads PROPERTIES LABELS "cpu")

add_test(NAME smoke_help COMMAND muscle -h)
set_tests_properties(smoke_help PROPERTIES
  LABELS "cpu"
  FAIL_REGULAR_EXPRESSION "Fatal error")

if(Python3_EXECUTABLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.py")
  add_test(NAME smoke_tiny_align
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.py"
            --muscle $<TARGET_FILE:muscle>
            --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tiny3.fa"
            --workdir "${CMAKE_BINARY_DIR}/smoke_tiny")
  set_tests_properties(smoke_tiny_align PROPERTIES LABELS "cpu")

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_data/fa/BB11001")
    add_test(NAME smoke_bb11001_2threads
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/smoke.py"
              --muscle $<TARGET_FILE:muscle>
              --input "${CMAKE_CURRENT_SOURCE_DIR}/test_data/fa/BB11001"
              --threads 2
              --workdir "${CMAKE_BINARY_DIR}/smoke_bb11001")
    set_tests_properties(smoke_bb11001_2threads PROPERTIES LABELS "cpu")
  endif()
endif()

add_test(NAME backend_contract_quick COMMAND backend_contract --suite quick)
set_tests_properties(backend_contract_quick PROPERTIES
  LABELS "cpu"
  SKIP_RETURN_CODE 77)

if(Python3_EXECUTABLE)
  # The batch backend must reproduce the reference bytes, and an explicit GPU
  # request without CUDA must fail loudly (both were reviewer-found gaps).
  set(MUSCLE_CLI_CUDA_ARGS)
  if(MUSCLE_CUDA)
    list(APPEND MUSCLE_CLI_CUDA_ARGS --expect-cuda)
  endif()
  add_test(NAME cli_cpu_batch_byte_equal
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_backend.py"
            --muscle $<TARGET_FILE:muscle>
            --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tiny3.fa"
            --workdir "${CMAKE_BINARY_DIR}/cli_backend" ${MUSCLE_CLI_CUDA_ARGS})
  set_tests_properties(cli_cpu_batch_byte_equal PROPERTIES LABELS "cpu")

  add_test(NAME oracle_dump_roundtrip
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/oracle_roundtrip.py"
            --oracle $<TARGET_FILE:pair_oracle>
            --compare "${CMAKE_CURRENT_SOURCE_DIR}/tests/compare_dumps.py"
            --workdir "${CMAKE_BINARY_DIR}/oracle_roundtrip")
  set_tests_properties(oracle_dump_roundtrip PROPERTIES LABELS "cpu")
endif()

add_test(NAME pair_oracle_tiny_cpu COMMAND pair_oracle --backend cpu --suite tiny)
set_tests_properties(pair_oracle_tiny_cpu PROPERTIES LABELS "cpu")

if(MUSCLE_CUDA)
  add_test(NAME pair_oracle_tiny_gpu COMMAND pair_oracle --backend gpu --suite tiny)
  set_tests_properties(pair_oracle_tiny_gpu PROPERTIES LABELS "gpu")
  add_test(NAME pair_oracle_threshold_gpu COMMAND pair_oracle --backend gpu --suite threshold)
  set_tests_properties(pair_oracle_threshold_gpu PROPERTIES LABELS "gpu")
  add_test(NAME backend_contract_gpu COMMAND backend_contract --suite gpu)
  set_tests_properties(backend_contract_gpu PROPERTIES
    LABELS "gpu"
    SKIP_RETURN_CODE 77)

  if(Python3_EXECUTABLE)
    add_test(NAME super5_backend_tiny_gpu
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/super5_backend.py"
              --binary $<TARGET_FILE:muscle_gpu>
              --input "${CMAKE_CURRENT_SOURCE_DIR}/test_data/fa/BB11001"
              --threads 1 --out "${CMAKE_BINARY_DIR}/super5_backend_tiny")
    set_tests_properties(super5_backend_tiny_gpu PROPERTIES LABELS "gpu" TIMEOUT 120)
    add_test(NAME gpu_integration
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/gpu_integration.py"
              --muscle $<TARGET_FILE:muscle>
              --input "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tiny3.fa"
              --reference "${CMAKE_BINARY_DIR}/cli_backend/reference.afa"
              --workdir "${CMAKE_BINARY_DIR}/gpu_integration")
    set_tests_properties(gpu_integration PROPERTIES
      LABELS "gpu"
      SKIP_RETURN_CODE 77
      DEPENDS cli_cpu_batch_byte_equal)
  endif()
endif()

# Record the configuration actually used; consumed by results/environment.json tooling.
file(WRITE "${CMAKE_BINARY_DIR}/muscle_build_config.txt"
  "build_type=${CMAKE_BUILD_TYPE}\n"
  "cxx_compiler=${CMAKE_CXX_COMPILER} ${CMAKE_CXX_COMPILER_VERSION}\n"
  "cxx_flags=${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}\n"
  "muscle_base_flags=${MUSCLE_BASE_FLAGS}\n"
  "sanitize=${MUSCLE_SANITIZE}\n"
  "sanitize_flags=${MUSCLE_SAN_FLAGS}\n"
  "strict_fp=${MUSCLE_STRICT_FP}\n"
  "cuda=${MUSCLE_CUDA}\n"
  "cuda_compiler=${CMAKE_CUDA_COMPILER} ${CMAKE_CUDA_COMPILER_VERSION}\n"
  "cuda_strict_flags=${MUSCLE_CUDA_STRICT_FLAGS}\n"
  "cuda_architectures=${CMAKE_CUDA_ARCHITECTURES}\n"
  "vcxproj_sha256=${MUSCLE_VCXPROJ_SHA256}\n"
  "gitver=${MUSCLE_GITVER}\n"
)
