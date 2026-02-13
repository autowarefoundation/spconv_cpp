set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Specify the cross-compiler
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# For CUDA cross-compilation (SBSA; covers Jetson Thor and other ARM64 CUDA platforms)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# Detect CUDA version from nvcc (CUDA 13.0+ uses sm_110 instead of sm_101)
execute_process(
    COMMAND ${CMAKE_CUDA_COMPILER} --version
    OUTPUT_VARIABLE _nvcc_out
    ERROR_VARIABLE _nvcc_err
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REGEX MATCH "release ([0-9]+)\\.([0-9]+)" _nvcc_match "${_nvcc_out}")
if(CMAKE_MATCH_1)
    set(_CUDA_VERSION_DETECTED "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
    message(STATUS "arm64-toolchain: detected CUDA version from nvcc: ${_CUDA_VERSION_DETECTED}")
else()
    message(FATAL_ERROR "arm64-toolchain.cmake: could not detect CUDA version from nvcc. Ensure ${CMAKE_CUDA_COMPILER} exists and returns a parseable --version (e.g. \"release 13.0\").")
endif()

set(CMAKE_CUDA_ARCHITECTURES "75;80;86;87;89;90;100;120" CACHE STRING "CUDA architectures for ARM64 (SBSA)")
if(_CUDA_VERSION_DETECTED VERSION_LESS "13.0")
    list(APPEND CMAKE_CUDA_ARCHITECTURES "101")
else()
    list(APPEND CMAKE_CUDA_ARCHITECTURES "110")
endif()

set(CUDA_TARGET_INCLUDE_PATH "/usr/local/cuda/targets/sbsa-linux/include")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CUDA_TARGET_INCLUDE_PATH}")

# Search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# Search for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Add cumm headers for spconv cross-compilation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_LIST_DIR}/../cumm/include")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_LIST_DIR}/../cumm/include")