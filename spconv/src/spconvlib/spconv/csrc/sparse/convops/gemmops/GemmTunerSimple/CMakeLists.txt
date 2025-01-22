
cmake_minimum_required(VERSION 3.20)
add_library(spconvlib_spconv_csrc_sparse_convops_gemmops_GemmTunerSimple OBJECT GemmTunerSimple_GemmTunerSimple.cc GemmTunerSimple_get_available_algo_str_from_arch.cc GemmTunerSimple_get_all_available.cc GemmTunerSimple_extract_mnk.cc GemmTunerSimple_extract_mnk_vector.cc GemmTunerSimple_cached_get_nvrtc_params.cc GemmTunerSimple_tune_and_cache.cc GemmTunerSimple_get_tuned_algo.cc GemmTunerSimple_run_with_tuned_result.cc)
# set_target_properties(spconvlib_spconv_csrc_sparse_convops_gemmops_GemmTunerSimple PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties( spconvlib_spconv_csrc_sparse_convops_gemmops_GemmTunerSimple PROPERTIES CXX_STANDARD 14 CUDA_STANDARD 14)

# set_target_properties(spconvlib_spconv_csrc_sparse_convops_gemmops_GemmTunerSimple PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(spconvlib_spconv_csrc_sparse_convops_gemmops_GemmTunerSimple PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(spconvlib_spconv_csrc_sparse_convops_gemmops_GemmTunerSimple PRIVATE
    /usr/local/cuda/include
    /home/kenzo/.local/lib/python3.10/site-packages/cumm/include)
target_compile_options(spconvlib_spconv_csrc_sparse_convops_gemmops_GemmTunerSimple PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
     -DTV_CUDA
    -gencode=arch=compute_75,code=sm_75
    -gencode=arch=compute_80,code=sm_80
    -gencode=arch=compute_86,code=sm_86
    -gencode=arch=compute_89,code=sm_89
    -Xcudafe "--diag_suppress=implicit_return_from_non_void_function"
    --expt-relaxed-constexpr
    -Xcompiler="-fPIC"
    -Xcompiler='-O3'>)
target_compile_options(spconvlib_spconv_csrc_sparse_convops_gemmops_GemmTunerSimple PRIVATE $<$<COMPILE_LANGUAGE:CXX>: 
    -DTV_CUDA
    -O3
    -fPIC>)

                