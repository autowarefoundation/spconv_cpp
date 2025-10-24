
cmake_minimum_required(VERSION 3.20)
add_library(spconvlib_spconv_csrc_sparse_all_ops_cpu2d_Point2VoxelCPU OBJECT Point2VoxelCPU_get_grid_size.cc Point2VoxelCPU_calc_meta_data.cc Point2VoxelCPU_Point2VoxelCPU.cc Point2VoxelCPU_point_to_voxel_static.cc Point2VoxelCPU_point_to_voxel_empty_mean_static.cc Point2VoxelCPU_point_to_voxel.cc Point2VoxelCPU_point_to_voxel_empty_mean.cc)
# set_target_properties(spconvlib_spconv_csrc_sparse_all_ops_cpu2d_Point2VoxelCPU PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties( spconvlib_spconv_csrc_sparse_all_ops_cpu2d_Point2VoxelCPU PROPERTIES CXX_STANDARD 14 CUDA_STANDARD 14)

# set_target_properties(spconvlib_spconv_csrc_sparse_all_ops_cpu2d_Point2VoxelCPU PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# set_target_properties(spconvlib_spconv_csrc_sparse_all_ops_cpu2d_Point2VoxelCPU PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(spconvlib_spconv_csrc_sparse_all_ops_cpu2d_Point2VoxelCPU PRIVATE
    /usr/local/cuda/include)
target_compile_options(spconvlib_spconv_csrc_sparse_all_ops_cpu2d_Point2VoxelCPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
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
target_compile_options(spconvlib_spconv_csrc_sparse_all_ops_cpu2d_Point2VoxelCPU PRIVATE $<$<COMPILE_LANGUAGE:CXX>: 
    -DTV_CUDA
    -O3
    -fPIC>)

                