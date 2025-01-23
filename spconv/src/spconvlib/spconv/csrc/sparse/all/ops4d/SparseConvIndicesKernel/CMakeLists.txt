
cmake_minimum_required(VERSION 3.20)
add_library(spconvlib_spconv_csrc_sparse_all_ops4d_SparseConvIndicesKernel OBJECT SparseConvIndicesKernel_calc_conv_indices_stage2_mask_output.cu SparseConvIndicesKernel_generate_conv_inds_stage1.cu SparseConvIndicesKernel_generate_conv_inds_stage1_5.cu SparseConvIndicesKernel_generate_conv_inds_stage2.cu SparseConvIndicesKernel_generate_conv_inds_mask_stage1.cu SparseConvIndicesKernel_generate_conv_inds_mask_stage1_direct_table.cu SparseConvIndicesKernel_generate_conv_inds_stage2_mask.cu SparseConvIndicesKernel_generate_conv_inds_stage2_mask_direct_table.cu SparseConvIndicesKernel_unique_and_assign_output_direct_hash.cu SparseConvIndicesKernel_unique_hash.cu SparseConvIndicesKernel_assign_output_direct_hash.cu SparseConvIndicesKernel_generate_subm_conv_inds.cu)
# set_target_properties(spconvlib_spconv_csrc_sparse_all_ops4d_SparseConvIndicesKernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties( spconvlib_spconv_csrc_sparse_all_ops4d_SparseConvIndicesKernel PROPERTIES CXX_STANDARD 14 CUDA_STANDARD 14)

# set_target_properties(spconvlib_spconv_csrc_sparse_all_ops4d_SparseConvIndicesKernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(spconvlib_spconv_csrc_sparse_all_ops4d_SparseConvIndicesKernel PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(spconvlib_spconv_csrc_sparse_all_ops4d_SparseConvIndicesKernel PRIVATE
    /usr/local/cuda/include)
target_compile_options(spconvlib_spconv_csrc_sparse_all_ops4d_SparseConvIndicesKernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
     -DTV_CUDA
    -gencode=arch=compute_75,code=sm_75
    -gencode=arch=compute_80,code=sm_80
    -gencode=arch=compute_86,code=sm_86
    -gencode=arch=compute_89,code=sm_89
    -Xcudafe "--diag_suppress=implicit_return_from_non_void_function"
    --expt-relaxed-constexpr
    -Xcompiler="-fPIC"
    -Xcompiler='-O3'>)
target_compile_options(spconvlib_spconv_csrc_sparse_all_ops4d_SparseConvIndicesKernel PRIVATE $<$<COMPILE_LANGUAGE:CXX>: 
    -DTV_CUDA
    -O3
    -fPIC>)

                