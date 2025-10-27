set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# Specify the cross-compiler
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# For CUDA cross-compilation
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

# Allow user to specify ARM64 variant: sbsa-linux (server) or aarch64-linux (Jetson)
set(ARM64_CUDA_TARGET "sbsa-linux" CACHE STRING "ARM64 CUDA target: sbsa-linux or aarch64-linux")
set_property(CACHE ARM64_CUDA_TARGET PROPERTY STRINGS "sbsa-linux" "aarch64-linux")

# Set architectures based on target
if(ARM64_CUDA_TARGET STREQUAL "aarch64-linux")
    # Jetson architectures
    set(CMAKE_CUDA_ARCHITECTURES "87;101" CACHE STRING "CUDA architectures for Jetson")
else()
    # Server/datacenter architectures
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;87;89;90;100;101;120" CACHE STRING "CUDA architectures for SBSA")
endif()

set(CUDA_TARGET_INCLUDE_PATH "/usr/local/cuda/targets/${ARM64_CUDA_TARGET}/include")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CUDA_TARGET_INCLUDE_PATH}")

# Search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# Search for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Add cumm headers for spconv cross-compilation
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${CMAKE_CURRENT_LIST_DIR}/../cumm/include")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -I${CMAKE_CURRENT_LIST_DIR}/../cumm/include")