
cmake_minimum_required(VERSION 3.20)
add_library(spconvlib_spconv_csrc_sparse_convops_ConvTuneResult OBJECT ConvTuneResult_ConvTuneResult.cc ConvTuneResult_is_valid.cc)
# set_target_properties(spconvlib_spconv_csrc_sparse_convops_ConvTuneResult PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties( spconvlib_spconv_csrc_sparse_convops_ConvTuneResult PROPERTIES CXX_STANDARD 14 CUDA_STANDARD 14)

# set_target_properties(spconvlib_spconv_csrc_sparse_convops_ConvTuneResult PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(spconvlib_spconv_csrc_sparse_convops_ConvTuneResult PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(spconvlib_spconv_csrc_sparse_convops_ConvTuneResult PRIVATE
    /usr/local/cuda/include
    /home/kenzo/.local/lib/python3.10/site-packages/cumm/include)
target_compile_options(spconvlib_spconv_csrc_sparse_convops_ConvTuneResult PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
     -DTV_CUDA
    -gencode=arch=compute_75,code=sm_75
    -gencode=arch=compute_80,code=sm_80
    -gencode=arch=compute_86,code=sm_86
    -gencode=arch=compute_89,code=sm_89
    -Xcudafe "--diag_suppress=implicit_return_from_non_void_function"
    --expt-relaxed-constexpr
    -Xcompiler="-fPIC"
    -Xcompiler='-O3'>)
target_compile_options(spconvlib_spconv_csrc_sparse_convops_ConvTuneResult PRIVATE $<$<COMPILE_LANGUAGE:CXX>: 
    -DTV_CUDA
    -O3
    -fPIC>)

                