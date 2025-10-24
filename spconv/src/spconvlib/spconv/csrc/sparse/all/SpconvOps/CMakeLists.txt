
cmake_minimum_required(VERSION 3.20)
add_library(spconvlib_spconv_csrc_sparse_all_SpconvOps OBJECT SpconvOps_cumm_version.cc SpconvOps_is_cpu_only_build.cc SpconvOps_pccm_version.cc SpconvOps_generate_conv_inds_stage1.cu SpconvOps_generate_conv_inds_stage1_5.cu SpconvOps_generate_conv_inds_stage2.cu SpconvOps_generate_conv_inds_mask_stage1.cu SpconvOps_generate_conv_inds_mask_stage1_direct_table.cu SpconvOps_unique_hash.cu SpconvOps_assign_output_direct_hash.cu SpconvOps_generate_conv_inds_mask_stage2.cu SpconvOps_generate_conv_inds_stage2_mask_direct_table.cu SpconvOps_generate_subm_conv_inds.cu SpconvOps_generate_conv_inds_cpu.cc SpconvOps_generate_subm_conv_inds_cpu.cc SpconvOps_maxpool_forward.cu SpconvOps_maxpool_backward.cu SpconvOps_indice_maxpool.cu SpconvOps_indice_maxpool_backward.cu SpconvOps_global_pool_rearrange.cu SpconvOps_maxpool_implicit_gemm_forward.cu SpconvOps_maxpool_implicit_gemm_backward.cu SpconvOps_avgpool_implicit_gemm_forward.cu SpconvOps_avgpool_implicit_gemm_backward.cu SpconvOps_maxpool_forward_cpu.cc SpconvOps_maxpool_backward_cpu.cc SpconvOps_gather_cpu.cc SpconvOps_scatter_add_cpu.cc SpconvOps_sort_1d_by_key_allocator.cu SpconvOps_sort_1d_by_key_allocator_v2.cu SpconvOps_sort_1d_by_key_split.cu SpconvOps_sort_1d_by_key_split_allocator.cu SpconvOps_sort_1d_by_key_split_allocator_v2.cu SpconvOps_count_bits.cu SpconvOps_reverse_bits.cu SpconvOps_maximum_value_int.cu SpconvOps_sort_1d_by_key.cu SpconvOps_calc_point2voxel_meta_data.cc SpconvOps_point2voxel_cpu.cc SpconvOps_point2voxel_cuda.cc SpconvOps_get_int32_max.cc SpconvOps_get_conv_output_size.cc SpconvOps_get_deconv_output_size.cc SpconvOps_apply_thrust_unique_to_indice_pairs_uniq.cu SpconvOps_get_handcrafted_max_act_out.cc SpconvOps_get_indice_gen_workspace_size.cc SpconvOps_get_indice_gen_tensors_from_workspace.cc SpconvOps_get_indice_pairs_implicit_gemm.cc SpconvOps_get_indice_pairs.cc)
# set_target_properties(spconvlib_spconv_csrc_sparse_all_SpconvOps PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties( spconvlib_spconv_csrc_sparse_all_SpconvOps PROPERTIES CXX_STANDARD 14 CUDA_STANDARD 14)

# set_target_properties(spconvlib_spconv_csrc_sparse_all_SpconvOps PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(spconvlib_spconv_csrc_sparse_all_SpconvOps PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(spconvlib_spconv_csrc_sparse_all_SpconvOps PRIVATE
    /usr/local/cuda/include)
target_compile_options(spconvlib_spconv_csrc_sparse_all_SpconvOps PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
     -DTV_CUDA
    -gencode=arch=compute_75,code=sm_75
    -gencode=arch=compute_80,code=sm_80
    -gencode=arch=compute_86,code=sm_86
    -gencode=arch=compute_87,code=sm_87
    -gencode=arch=compute_89,code=sm_89
    -gencode=arch=compute_90,code=sm_90
    -gencode=arch=compute_100,code=sm_100
    -gencode=arch=compute_120,code=sm_120
    -Xcudafe "--diag_suppress=implicit_return_from_non_void_function"
    -Xcompiler -fno-gnu-unique
    -Xcompiler -fvisibility=hidden
    --expt-relaxed-constexpr
    -Xcompiler="-fPIC"
    -Xcompiler='-O3'>)
target_compile_options(spconvlib_spconv_csrc_sparse_all_SpconvOps PRIVATE $<$<COMPILE_LANGUAGE:CXX>: 
    -DTV_CUDA
    -O3
    -fPIC>)

                