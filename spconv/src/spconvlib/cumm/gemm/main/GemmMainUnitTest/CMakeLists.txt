
cmake_minimum_required(VERSION 3.20)
add_library(spconvlib_cumm_gemm_main_GemmMainUnitTest OBJECT GemmMainUnitTest_matmul_split_Simt_f32f32f32_1.cu GemmMainUnitTest_matmul_split_Simt_f16f16f16_1.cu GemmMainUnitTest_matmul_split_Volta_f16f16f16_1.cu GemmMainUnitTest_matmul_split_Turing_f16f16f16_1.cu GemmMainUnitTest_get_all_algo_desp.cc GemmMainUnitTest_extract_mnk.cc GemmMainUnitTest_align_to_power2.cc GemmMainUnitTest_device_synchronize.cc GemmMainUnitTest_stream_synchronize.cc GemmMainUnitTest_simple_select_tile_shape.cc GemmMainUnitTest_matmul2.cu)
# set_target_properties(spconvlib_cumm_gemm_main_GemmMainUnitTest PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties( spconvlib_cumm_gemm_main_GemmMainUnitTest PROPERTIES CXX_STANDARD 14 CUDA_STANDARD 14)

# set_target_properties(spconvlib_cumm_gemm_main_GemmMainUnitTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(spconvlib_cumm_gemm_main_GemmMainUnitTest PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(spconvlib_cumm_gemm_main_GemmMainUnitTest PRIVATE
    /usr/local/cuda/include
    /home/kenzo/.local/lib/python3.10/site-packages/cumm/include)
target_compile_options(spconvlib_cumm_gemm_main_GemmMainUnitTest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
     -DTV_CUDA
    -gencode=arch=compute_75,code=sm_75
    -gencode=arch=compute_80,code=sm_80
    -gencode=arch=compute_86,code=sm_86
    -gencode=arch=compute_89,code=sm_89
    -Xcudafe "--diag_suppress=implicit_return_from_non_void_function"
    --expt-relaxed-constexpr
    -Xcompiler="-fPIC"
    -Xcompiler='-O3'>)
target_compile_options(spconvlib_cumm_gemm_main_GemmMainUnitTest PRIVATE $<$<COMPILE_LANGUAGE:CXX>: 
    -DTV_CUDA
    -O3
    -fPIC>)

                