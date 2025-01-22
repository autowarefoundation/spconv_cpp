
cmake_minimum_required(VERSION 3.20)
add_library(spconvlib_cumm_common_CompileInfo OBJECT CompileInfo_get_compiled_cuda_version.cu CompileInfo_get_compiled_cuda_arch.cc CompileInfo_get_compiled_gemm_cuda_arch.cc CompileInfo_arch_is_compiled.cc CompileInfo_arch_is_compiled_gemm.cc CompileInfo_arch_is_compatible.cc CompileInfo_arch_is_compatible_gemm.cc CompileInfo_algo_can_use_ptx.cc CompileInfo_gemm_algo_can_use_ptx.cc CompileInfo_algo_can_be_nvrtc_compiled.cc)
# set_target_properties(spconvlib_cumm_common_CompileInfo PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties( spconvlib_cumm_common_CompileInfo PROPERTIES CXX_STANDARD 14 CUDA_STANDARD 14)

# set_target_properties(spconvlib_cumm_common_CompileInfo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(spconvlib_cumm_common_CompileInfo PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(spconvlib_cumm_common_CompileInfo PRIVATE
)
target_compile_options(spconvlib_cumm_common_CompileInfo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
     --expt-relaxed-constexpr
    -Xcompiler="-fPIC"
    -Xcompiler='-O3'>)
target_compile_options(spconvlib_cumm_common_CompileInfo PRIVATE $<$<COMPILE_LANGUAGE:CXX>: 
    -O3
    -fPIC>)

                