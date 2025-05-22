
cmake_minimum_required(VERSION 3.20)
add_library(spconvlib_cumm_conv_main_ConvMainUnitTest OBJECT ConvMainUnitTest_matmul_split_Simt_f32f32f32_0.cu ConvMainUnitTest_matmul_split_Simt_f16f16f16_0.cu ConvMainUnitTest_matmul_split_Volta_f16f16f16_0.cu ConvMainUnitTest_matmul_split_Turing_s8s8s8_0.cu ConvMainUnitTest_matmul_split_Turing_s8s8f32_0.cu ConvMainUnitTest_matmul_split_Turing_s8s8f16_0.cu ConvMainUnitTest_matmul_split_Turing_f16f16f16_0.cu ConvMainUnitTest_matmul_split_Ampere_f32f32f32_0.cu ConvMainUnitTest_matmul_split_Ampere_f16f16f16_0.cu ConvMainUnitTest_matmul_split_Ampere_s8s8s8_0.cu ConvMainUnitTest_matmul_split_Ampere_s8s8f32_0.cu ConvMainUnitTest_matmul_split_Ampere_s8s8f16_0.cu ConvMainUnitTest_extract_mnk.cc ConvMainUnitTest_implicit_gemm2.cu ConvMainUnitTest_get_all_conv_algo_desp.cc)
# set_target_properties(spconvlib_cumm_conv_main_ConvMainUnitTest PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties( spconvlib_cumm_conv_main_ConvMainUnitTest PROPERTIES CXX_STANDARD 14 CUDA_STANDARD 14)

# set_target_properties(spconvlib_cumm_conv_main_ConvMainUnitTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(spconvlib_cumm_conv_main_ConvMainUnitTest PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(spconvlib_cumm_conv_main_ConvMainUnitTest PRIVATE
    /usr/local/cuda/include)
target_compile_options(spconvlib_cumm_conv_main_ConvMainUnitTest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
     -DTV_CUDA
    -gencode=arch=compute_75,code=sm_75
    -gencode=arch=compute_80,code=sm_80
    -gencode=arch=compute_86,code=sm_86
    -gencode=arch=compute_89,code=sm_89
    -gencode=arch=compute_120,code=sm_120
    -Xcudafe "--diag_suppress=implicit_return_from_non_void_function"
    --expt-relaxed-constexpr
    -Xcompiler="-fPIC"
    -Xcompiler='-O3'>)
target_compile_options(spconvlib_cumm_conv_main_ConvMainUnitTest PRIVATE $<$<COMPILE_LANGUAGE:CXX>: 
    -DTV_CUDA
    -O3
    -fPIC>)

                