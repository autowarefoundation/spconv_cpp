
cmake_minimum_required(VERSION 3.20)
add_library(spconvlib_spconv_csrc_sparse_convops_convops_ConvTunerSimple OBJECT ConvTunerSimple_ConvTunerSimple.cc ConvTunerSimple_get_available_algo_str_from_arch.cc ConvTunerSimple_get_all_available.cc ConvTunerSimple_cached_get_nvrtc_params.cc ConvTunerSimple_tune_and_cache.cc ConvTunerSimple_get_tuned_algo.cc ConvTunerSimple_run_with_tuned_result.cc ConvTunerSimple_query_workspace_size.cc)
# set_target_properties(spconvlib_spconv_csrc_sparse_convops_convops_ConvTunerSimple PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties( spconvlib_spconv_csrc_sparse_convops_convops_ConvTunerSimple PROPERTIES CXX_STANDARD 14 CUDA_STANDARD 14)

# set_target_properties(spconvlib_spconv_csrc_sparse_convops_convops_ConvTunerSimple PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(spconvlib_spconv_csrc_sparse_convops_convops_ConvTunerSimple PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(spconvlib_spconv_csrc_sparse_convops_convops_ConvTunerSimple PRIVATE
    /usr/local/cuda/include)
target_compile_options(spconvlib_spconv_csrc_sparse_convops_convops_ConvTunerSimple PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
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
target_compile_options(spconvlib_spconv_csrc_sparse_convops_convops_ConvTunerSimple PRIVATE $<$<COMPILE_LANGUAGE:CXX>: 
    -DTV_CUDA
    -O3
    -fPIC>)

                