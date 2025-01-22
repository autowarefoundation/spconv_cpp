
cmake_minimum_required(VERSION 3.20)
add_library(spconvlib_cumm_conv_main_cpAmpere_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T16816_200_C301LLL_SK_S8_ConvParams OBJECT ConvParams_ConvParams.cu)
# set_target_properties(spconvlib_cumm_conv_main_cpAmpere_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T16816_200_C301LLL_SK_S8_ConvParams PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties( spconvlib_cumm_conv_main_cpAmpere_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T16816_200_C301LLL_SK_S8_ConvParams PROPERTIES CXX_STANDARD 14 CUDA_STANDARD 14)

# set_target_properties(spconvlib_cumm_conv_main_cpAmpere_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T16816_200_C301LLL_SK_S8_ConvParams PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(spconvlib_cumm_conv_main_cpAmpere_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T16816_200_C301LLL_SK_S8_ConvParams PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(spconvlib_cumm_conv_main_cpAmpere_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T16816_200_C301LLL_SK_S8_ConvParams PRIVATE
    /usr/local/cuda/include
    /home/kenzo/.local/lib/python3.10/site-packages/cumm/include)
target_compile_options(spconvlib_cumm_conv_main_cpAmpere_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T16816_200_C301LLL_SK_S8_ConvParams PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
     -DTV_CUDA
    --expt-relaxed-constexpr
    -Xcompiler="-fPIC"
    -Xcompiler='-O3'>)
target_compile_options(spconvlib_cumm_conv_main_cpAmpere_s8s8s8s32f16tnt_m64n32k32m32n32k32A1T16816_200_C301LLL_SK_S8_ConvParams PRIVATE $<$<COMPILE_LANGUAGE:CXX>: 
    -O3
    -fPIC>)

                