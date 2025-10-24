
cmake_minimum_required(VERSION 3.20)
add_library(spconvlib_cumm_gemm_main_Simt_f32f32f32f32f32tnt_m128n64k8m64n32k8A1_200_S1_GemmKernel OBJECT GemmKernel_gemm_kernel.cu)
# set_target_properties(spconvlib_cumm_gemm_main_Simt_f32f32f32f32f32tnt_m128n64k8m64n32k8A1_200_S1_GemmKernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties( spconvlib_cumm_gemm_main_Simt_f32f32f32f32f32tnt_m128n64k8m64n32k8A1_200_S1_GemmKernel PROPERTIES CXX_STANDARD 14 CUDA_STANDARD 14)

# set_target_properties(spconvlib_cumm_gemm_main_Simt_f32f32f32f32f32tnt_m128n64k8m64n32k8A1_200_S1_GemmKernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(spconvlib_cumm_gemm_main_Simt_f32f32f32f32f32tnt_m128n64k8m64n32k8A1_200_S1_GemmKernel PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(spconvlib_cumm_gemm_main_Simt_f32f32f32f32f32tnt_m128n64k8m64n32k8A1_200_S1_GemmKernel PRIVATE
    /usr/local/cuda/include)
target_compile_options(spconvlib_cumm_gemm_main_Simt_f32f32f32f32f32tnt_m128n64k8m64n32k8A1_200_S1_GemmKernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
     -DTV_CUDA
    -gencode=arch=compute_75,code=sm_75
    -gencode=arch=compute_80,code=sm_80
    -gencode=arch=compute_86,code=sm_86
    -gencode=arch=compute_87,code=sm_87
    -gencode=arch=compute_89,code=sm_89
    -gencode=arch=compute_90,code=sm_90
    -gencode=arch=compute_100,code=sm_100
    -gencode=arch=compute_101,code=sm_101
    -gencode=arch=compute_120,code=sm_120
    -Xcudafe "--diag_suppress=implicit_return_from_non_void_function"
    --expt-relaxed-constexpr
    -Xcompiler="-fPIC"
    -Xcompiler='-O3'>)
target_compile_options(spconvlib_cumm_gemm_main_Simt_f32f32f32f32f32tnt_m128n64k8m64n32k8A1_200_S1_GemmKernel PRIVATE $<$<COMPILE_LANGUAGE:CXX>: 
    -O3
    -fPIC>)

                