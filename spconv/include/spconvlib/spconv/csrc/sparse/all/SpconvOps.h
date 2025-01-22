#pragma once
#include <spconvlib/spconv/csrc/sparse/all/ThrustCustomAllocatorV2.h>
#include <spconvlib/spconv/csrc/sparse/alloc/ExternalAllocator.h>
#include <spconvlib/cumm/common/GemmBasicHost.h>
#include <spconvlib/spconv/csrc/sparse/alloc/ThrustAllocator.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu1d/Point2VoxelCPU.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu1d/SparseConvIndicesCPU.h>
#include <spconvlib/spconv/csrc/sparse/all/ops1d/Point2Voxel.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu2d/Point2VoxelCPU.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu2d/SparseConvIndicesCPU.h>
#include <spconvlib/spconv/csrc/sparse/all/ops2d/Point2Voxel.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu3d/Point2VoxelCPU.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu3d/SparseConvIndicesCPU.h>
#include <spconvlib/spconv/csrc/sparse/all/ops3d/Point2Voxel.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu4d/Point2VoxelCPU.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu4d/SparseConvIndicesCPU.h>
#include <spconvlib/spconv/csrc/sparse/all/ops4d/Point2Voxel.h>
#define SPCONV_ALLOC_D_FILTERS "DFilters"
#define SPCONV_ALLOC_D_IN "DIn"
#define SPCONV_ALLOC_FEATURES "Features"
#define SPCONV_ALLOC_FILTERS "Filters"
#define SPCONV_ALLOC_HASH_K_OR_KV "HashKOrKV"
#define SPCONV_ALLOC_HASH_V "HashV"
#define SPCONV_ALLOC_INDICE_NUM_PER_LOC "IndiceNumPerLoc"
#define SPCONV_ALLOC_INDICE_PAIRS_UNIQ "IndicePairsUniq"
#define SPCONV_ALLOC_INDICE_PAIRS_UNIQ_BACKUP "IndicePairsUniqBackup"
#define SPCONV_ALLOC_INP_BUFFER "InpBuffer"
#define SPCONV_ALLOC_MASK_ARG_SORT "MaskArgSort"
#define SPCONV_ALLOC_MASK_ARG_SORT_BWD "MaskArgSortBwd"
#define SPCONV_ALLOC_MASK_OUTPUT_FWD "MaskOutputFwd"
#define SPCONV_ALLOC_OUT_BP "OutBp"
#define SPCONV_ALLOC_OUT_BUFFER "OutBuffer"
#define SPCONV_ALLOC_OUT_FEATURES "OutFeatures"
#define SPCONV_ALLOC_OUT_INDICES "OutIndices"
#define SPCONV_ALLOC_PAIR_BWD "PairBwd"
#define SPCONV_ALLOC_PAIR_FWD "PairFwd"
#define SPCONV_ALLOC_PAIR_MASK "PairMask"
#define SPCONV_ALLOC_PAIR_MASK_BWD "PairMaskBwd"
#define SPCONV_ALLOC_THRUST_TEMP "ThrustTemp"
#define SPCONV_ALLOC_TIGHT_UNIQUE_COUNT "TightUniqueCount"
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
using ThrustCustomAllocatorV2 = spconvlib::spconv::csrc::sparse::all::ThrustCustomAllocatorV2;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using ThrustAllocator = spconvlib::spconv::csrc::sparse::alloc::ThrustAllocator;
using Point2Voxel1DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu1d::Point2VoxelCPU;
using SpconvIndicesCPU1D = spconvlib::spconv::csrc::sparse::all::ops_cpu1d::SparseConvIndicesCPU;
using Point2Voxel1D = spconvlib::spconv::csrc::sparse::all::ops1d::Point2Voxel;
using Point2Voxel2DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu2d::Point2VoxelCPU;
using SpconvIndicesCPU2D = spconvlib::spconv::csrc::sparse::all::ops_cpu2d::SparseConvIndicesCPU;
using Point2Voxel2D = spconvlib::spconv::csrc::sparse::all::ops2d::Point2Voxel;
using Point2Voxel3DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu3d::Point2VoxelCPU;
using SpconvIndicesCPU3D = spconvlib::spconv::csrc::sparse::all::ops_cpu3d::SparseConvIndicesCPU;
using Point2Voxel3D = spconvlib::spconv::csrc::sparse::all::ops3d::Point2Voxel;
using Point2Voxel4DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::Point2VoxelCPU;
using SpconvIndicesCPU4D = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::SparseConvIndicesCPU;
using Point2Voxel4D = spconvlib::spconv::csrc::sparse::all::ops4d::Point2Voxel;
struct SpconvOps {
  /**
   * get cumm version when build spconv.
   *         
   */
  static std::string cumm_version();
  
  static bool is_cpu_only_build();
  /**
   * get pccm version when build spconv.
   *         
   */
  static std::string pccm_version();
  /**
   * @param indices 
   * @param indice_pairs 
   * @param indice_pairs_uniq 
   * @param indice_num_per_loc 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param transposed 
   * @param stream_int 
   */
  static void generate_conv_inds_stage1(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor indice_pairs_uniq, tv::Tensor indice_num_per_loc, int batch_size, std::vector<int> output_dims, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
  /**
   * @param indice_pairs_uniq 
   * @param ndim 
   * @param uniq_size 
   * @param stream_int 
   */
  static int generate_conv_inds_stage1_5(tv::Tensor indice_pairs_uniq, int ndim, int64_t uniq_size, std::uintptr_t stream_int = 0);
  /**
   * @param indices 
   * @param hashdata_k 
   * @param hashdata_v 
   * @param indice_pairs 
   * @param indice_pairs_uniq 
   * @param indice_pairs_uniq_before_sort 
   * @param out_inds 
   * @param indice_num_per_loc 
   * @param num_out_act 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param transposed 
   * @param stream_int 
   * @param use_bound_algo 
   */
  static int generate_conv_inds_stage2(tv::Tensor indices, tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor indice_pairs, tv::Tensor indice_pairs_uniq, tv::Tensor indice_pairs_uniq_before_sort, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int num_out_act, int batch_size, std::vector<int> output_dims, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, bool transposed = false, std::uintptr_t stream_int = 0, bool use_bound_algo = false);
  /**
   * @param indices 
   * @param indice_pairs_bwd 
   * @param indice_pairs_uniq 
   * @param indice_num_per_loc 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param transposed 
   * @param stream_int 
   */
  static void generate_conv_inds_mask_stage1(tv::Tensor indices, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor indice_num_per_loc, int batch_size, std::vector<int> output_dims, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
  /**
   * @param indices 
   * @param hashdata_k 
   * @param hashdata_v 
   * @param indice_pairs_bwd 
   * @param indice_pairs_uniq 
   * @param indice_num_per_loc 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param transposed 
   * @param stream_int 
   */
  static void generate_conv_inds_mask_stage1_direct_table(tv::Tensor indices, tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor indice_num_per_loc, int batch_size, std::vector<int> output_dims, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
  /**
   * @param hashdata_k 
   * @param hashdata_v 
   * @param uniq_cnt 
   * @param out_indices_offset 
   * @param num_out_bound 
   * @param stream_int 
   */
  static int unique_hash(tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor uniq_cnt, tv::Tensor out_indices_offset, int num_out_bound, std::uintptr_t stream_int = 0);
  /**
   * @param out_indices_offset 
   * @param out_indices 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param stream_int 
   */
  static void assign_output_direct_hash(tv::Tensor out_indices_offset, tv::Tensor out_indices, int batch_size, std::vector<int> output_dims, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, std::uintptr_t stream_int = 0);
  /**
   * @param indices 
   * @param hashdata_k 
   * @param hashdata_v 
   * @param indice_pairs_fwd 
   * @param indice_pairs_bwd 
   * @param indice_pairs_uniq 
   * @param indice_pairs_uniq_before_sort 
   * @param out_inds 
   * @param mask_fwd 
   * @param mask_bwd 
   * @param num_out_act 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param transposed 
   * @param stream_int 
   */
  static int generate_conv_inds_mask_stage2(tv::Tensor indices, tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor indice_pairs_fwd, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor indice_pairs_uniq_before_sort, tv::Tensor out_inds, tv::Tensor mask_fwd, tv::Tensor mask_bwd, int num_out_act, int batch_size, std::vector<int> output_dims, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
  /**
   * @param indices 
   * @param hashdata_k 
   * @param hashdata_v 
   * @param indice_pairs_fwd 
   * @param indice_pairs_bwd 
   * @param indice_pairs_uniq 
   * @param indice_pairs_uniq_before_sort 
   * @param out_inds 
   * @param mask_fwd 
   * @param mask_bwd 
   * @param num_out_act 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param transposed 
   * @param stream_int 
   */
  static int generate_conv_inds_stage2_mask_direct_table(tv::Tensor indices, tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor indice_pairs_fwd, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor indice_pairs_uniq_before_sort, tv::Tensor out_inds, tv::Tensor mask_fwd, tv::Tensor mask_bwd, int num_out_act, int batch_size, std::vector<int> output_dims, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
  /**
   * @param indices 
   * @param hashdata_k 
   * @param hashdata_v 
   * @param indice_pairs 
   * @param out_inds 
   * @param indice_num_per_loc 
   * @param batch_size 
   * @param input_dims 
   * @param ksize 
   * @param dilation 
   * @param indice_pair_mask 
   * @param backward 
   * @param stream_int 
   */
  static int generate_subm_conv_inds(tv::Tensor indices, tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor indice_pairs, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> dilation, tv::Tensor indice_pair_mask = tv::Tensor(), bool backward = false, std::uintptr_t stream_int = 0);
  /**
   * @param indices 
   * @param indice_pairs 
   * @param out_inds 
   * @param indice_num_per_loc 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param transposed 
   */
  static int generate_conv_inds_cpu(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size, std::vector<int> output_dims, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, bool transposed = false);
  /**
   * @param indices 
   * @param indice_pairs 
   * @param out_inds 
   * @param indice_num_per_loc 
   * @param batch_size 
   * @param input_dims 
   * @param ksize 
   * @param dilation 
   */
  static int generate_subm_conv_inds_cpu(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size, std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> dilation);
  /**
   * @param out 
   * @param inp 
   * @param out_inds 
   * @param in_inds 
   * @param stream 
   */
  static void maxpool_forward(tv::Tensor out, tv::Tensor inp, tv::Tensor out_inds, tv::Tensor in_inds, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param inp 
   * @param dout 
   * @param dinp 
   * @param out_inds 
   * @param in_inds 
   * @param stream 
   */
  static void maxpool_backward(tv::Tensor out, tv::Tensor inp, tv::Tensor dout, tv::Tensor dinp, tv::Tensor out_inds, tv::Tensor in_inds, std::uintptr_t stream = 0);
  /**
   * @param out_features 
   * @param features 
   * @param indice_pairs 
   * @param indice_pair_num 
   * @param num_activate_out 
   * @param stream 
   */
  static void indice_maxpool(tv::Tensor out_features, tv::Tensor features, tv::Tensor indice_pairs, tv::Tensor indice_pair_num, int num_activate_out, std::uintptr_t stream = 0);
  /**
   * @param din 
   * @param features 
   * @param out_features 
   * @param out_bp 
   * @param indice_pairs 
   * @param indice_pair_num 
   * @param stream 
   */
  static void indice_maxpool_backward(tv::Tensor din, tv::Tensor features, tv::Tensor out_features, tv::Tensor out_bp, tv::Tensor indice_pairs, tv::Tensor indice_pair_num, std::uintptr_t stream = 0);
  /**
   * @param out_indices 
   * @param coords 
   * @param counts 
   * @param stream 
   */
  static void global_pool_rearrange(tv::Tensor out_indices, tv::Tensor coords, tv::Tensor counts, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param inp 
   * @param inds 
   * @param stream 
   */
  static void maxpool_implicit_gemm_forward(tv::Tensor out, tv::Tensor inp, tv::Tensor inds, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param inp 
   * @param dout 
   * @param dinp 
   * @param inds 
   * @param stream 
   */
  static void maxpool_implicit_gemm_backward(tv::Tensor out, tv::Tensor inp, tv::Tensor dout, tv::Tensor dinp, tv::Tensor inds, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param inp 
   * @param inds 
   * @param count_out 
   * @param stream 
   */
  static void avgpool_implicit_gemm_forward(tv::Tensor out, tv::Tensor inp, tv::Tensor inds, tv::Tensor count_out, std::uintptr_t stream = 0);
  /**
   * @param dout 
   * @param dinp 
   * @param inds 
   * @param count_out 
   * @param stream 
   */
  static void avgpool_implicit_gemm_backward(tv::Tensor dout, tv::Tensor dinp, tv::Tensor inds, tv::Tensor count_out, std::uintptr_t stream = 0);
  /**
   * @param out 
   * @param inp 
   * @param out_inds 
   * @param in_inds 
   */
  static void maxpool_forward_cpu(tv::Tensor out, tv::Tensor inp, tv::Tensor out_inds, tv::Tensor in_inds);
  /**
   * @param out 
   * @param inp 
   * @param dout 
   * @param dinp 
   * @param out_inds 
   * @param in_inds 
   */
  static void maxpool_backward_cpu(tv::Tensor out, tv::Tensor inp, tv::Tensor dout, tv::Tensor dinp, tv::Tensor out_inds, tv::Tensor in_inds);
  /**
   * @param out 
   * @param inp 
   * @param inds 
   */
  static void gather_cpu(tv::Tensor out, tv::Tensor inp, tv::Tensor inds);
  /**
   * @param out 
   * @param inp 
   * @param inds 
   */
  static void scatter_add_cpu(tv::Tensor out, tv::Tensor inp, tv::Tensor inds);
  /**
   * @param data 
   * @param alloc_func 
   * @param indices 
   * @param stream 
   * @param mask_count 
   * @param do_sort 
   */
  static tv::Tensor sort_1d_by_key_allocator(tv::Tensor data, std::function<std::uintptr_t(std::size_t)> alloc_func, tv::Tensor indices = tv::Tensor(), std::uintptr_t stream = 0, int mask_count = 1, bool do_sort = true);
  /**
   * @param data 
   * @param allocator 
   * @param indices 
   * @param stream 
   * @param mask_count 
   * @param do_sort 
   */
  static tv::Tensor sort_1d_by_key_allocator_v2(tv::Tensor data, ThrustAllocator& allocator, tv::Tensor indices = tv::Tensor(), std::uintptr_t stream = 0, int mask_count = 1, bool do_sort = true);
  /**
   * @param data 
   * @param mask 
   * @param indices 
   * @param stream 
   * @param mask_output 
   */
  static tv::Tensor sort_1d_by_key_split(tv::Tensor data, tv::Tensor mask, tv::Tensor indices = tv::Tensor(), std::uintptr_t stream = 0, bool mask_output = false);
  /**
   * @param data 
   * @param alloc_func 
   * @param mask 
   * @param indices 
   * @param stream 
   * @param mask_output 
   */
  static tv::Tensor sort_1d_by_key_split_allocator(tv::Tensor data, std::function<std::uintptr_t(std::size_t)> alloc_func, tv::Tensor mask, tv::Tensor indices = tv::Tensor(), std::uintptr_t stream = 0, bool mask_output = false);
  /**
   * @param data 
   * @param allocator 
   * @param mask 
   * @param indices 
   * @param stream 
   * @param mask_output 
   */
  static tv::Tensor sort_1d_by_key_split_allocator_v2(tv::Tensor data, ThrustAllocator& allocator, tv::Tensor mask, tv::Tensor indices = tv::Tensor(), std::uintptr_t stream = 0, bool mask_output = false);
  /**
   * @param a 
   */
  static tv::Tensor count_bits(tv::Tensor a);
  /**
   * @param a 
   */
  static tv::Tensor reverse_bits(tv::Tensor a);
  /**
   * @param data 
   * @param value 
   * @param stream_int 
   */
  static void maximum_value_int(tv::Tensor data, int value, std::uintptr_t stream_int);
  /**
   * @param data 
   * @param indices 
   * @param stream 
   */
  static tv::Tensor sort_1d_by_key(tv::Tensor data, tv::Tensor indices = tv::Tensor(), std::uintptr_t stream = 0);
  /**
   * @param vsize_xyz 
   * @param coors_range_xyz 
   */
  static std::tuple<std::vector<float>, std::vector<int>, std::vector<int>, std::vector<float>> calc_point2voxel_meta_data(std::vector<float> vsize_xyz, std::vector<float> coors_range_xyz);
  /**
   * @param points 
   * @param voxels 
   * @param indices 
   * @param num_per_voxel 
   * @param densehashdata 
   * @param pc_voxel_id 
   * @param vsize 
   * @param grid_size 
   * @param grid_stride 
   * @param coors_range 
   * @param empty_mean 
   * @param clear_voxels 
   */
  static std::tuple<tv::Tensor, tv::Tensor, tv::Tensor> point2voxel_cpu(tv::Tensor points, tv::Tensor voxels, tv::Tensor indices, tv::Tensor num_per_voxel, tv::Tensor densehashdata, tv::Tensor pc_voxel_id, std::vector<float> vsize, std::vector<int> grid_size, std::vector<int> grid_stride, std::vector<float> coors_range, bool empty_mean = false, bool clear_voxels = true);
  /**
   * @param points 
   * @param voxels 
   * @param indices 
   * @param num_per_voxel 
   * @param hashdata 
   * @param point_indice_data 
   * @param pc_voxel_id 
   * @param vsize 
   * @param grid_size 
   * @param grid_stride 
   * @param coors_range 
   * @param empty_mean 
   * @param clear_voxels 
   * @param stream_int 
   */
  static std::tuple<tv::Tensor, tv::Tensor, tv::Tensor> point2voxel_cuda(tv::Tensor points, tv::Tensor voxels, tv::Tensor indices, tv::Tensor num_per_voxel, tv::Tensor hashdata, tv::Tensor point_indice_data, tv::Tensor pc_voxel_id, std::vector<float> vsize, std::vector<int> grid_size, std::vector<int64_t> grid_stride, std::vector<float> coors_range, bool empty_mean = false, bool clear_voxels = true, std::uintptr_t stream_int = 0);
  
  static int get_int32_max();
  /**
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   */
  static std::vector<int> get_conv_output_size(std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation);
  /**
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param output_padding 
   */
  static std::vector<int> get_deconv_output_size(std::vector<int> input_dims, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, std::vector<int> output_padding);
  /**
   * @param data 
   * @param allocator 
   * @param stream_int 
   */
  static int apply_thrust_unique_to_indice_pairs_uniq(tv::Tensor data, ThrustAllocator& allocator, std::uintptr_t stream_int = 0);
  /**
   * @param num_act_in 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   */
  static int get_handcrafted_max_act_out(size_t num_act_in, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation);
  /**
   * @param kv 
   * @param num_act_in 
   * @param num_act_out_bound 
   * @param max_act_out_in_theory 
   * @param subm 
   * @param use_int64_hash_k 
   * @param direct_table 
   */
  static std::size_t get_indice_gen_workspace_size(size_t kv, size_t num_act_in, size_t num_act_out_bound, size_t max_act_out_in_theory, bool subm, bool use_int64_hash_k, bool direct_table);
  /**
   * @param workspace 
   * @param kv 
   * @param num_act_in 
   * @param num_act_out_bound 
   * @param max_act_out_in_theory 
   * @param subm 
   * @param use_int64_hash_k 
   * @param direct_table 
   */
  static std::unordered_map<std::string, tv::Tensor> get_indice_gen_tensors_from_workspace(uint8_t* workspace, size_t kv, size_t num_act_in, size_t num_act_out_bound, size_t max_act_out_in_theory, bool subm, bool use_int64_hash_k, bool direct_table);
  /**
   * @param allocator 
   * @param indices 
   * @param batch_size 
   * @param input_dims 
   * @param algo 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param out_padding 
   * @param subm 
   * @param transposed 
   * @param is_train 
   * @param stream_int 
   * @param num_out_act_bound 
   * @param timer 
   * @param direct_table 
   * @param do_sort 
   * @param preallocated 
   */
  static std::tuple<tv::Tensor, int> get_indice_pairs_implicit_gemm(ExternalAllocator& allocator, tv::Tensor indices, int batch_size, std::vector<int> input_dims, int algo, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, std::vector<int> out_padding, bool subm, bool transposed, bool is_train, std::uintptr_t stream_int = 0, int num_out_act_bound = -1, tv::CUDAKernelTimer timer = tv::CUDAKernelTimer(false), bool direct_table = false, bool do_sort = true, std::unordered_map<std::string, tv::Tensor> preallocated = std::unordered_map<std::string, tv::Tensor>{});
  /**
   * @param allocator 
   * @param indices 
   * @param batch_size 
   * @param input_dims 
   * @param algo 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param out_padding 
   * @param subm 
   * @param transposed 
   * @param stream_int 
   * @param num_out_act_bound 
   * @param num_input_act_bound 
   */
  static int get_indice_pairs(ExternalAllocator& allocator, tv::Tensor indices, int batch_size, std::vector<int> input_dims, int algo, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, std::vector<int> out_padding, bool subm, bool transposed, std::uintptr_t stream_int = 0, int num_out_act_bound = -1, int num_input_act_bound = -1);
};
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib