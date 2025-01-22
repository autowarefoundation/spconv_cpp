#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
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
int SpconvOps::get_indice_pairs(ExternalAllocator& allocator, tv::Tensor indices, int batch_size, std::vector<int> input_dims, int algo, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, std::vector<int> out_padding, bool subm, bool transposed, std::uintptr_t stream_int, int num_out_act_bound, int num_input_act_bound)   {
  
  int kv = std::accumulate(ksize.begin(), ksize.end(), 1, std::multiplies<int>());
  auto conv_algo = static_cast<tv::gemm::SparseConvAlgo>(algo);
  TV_ASSERT_RT_ERR(conv_algo == tv::gemm::SparseConvAlgo::kNative, "only support kNative");
  if (num_out_act_bound > 0){
      TV_ASSERT_RT_ERR(num_input_act_bound > 0 && indices.dim(0) <= num_input_act_bound, 
          "out bound and input bound must both larger than zero");
  }
  std::vector<int> out_shape;
  if (!subm){
      if (transposed){
          out_shape = get_deconv_output_size(input_dims, ksize, stride, padding, dilation, out_padding);
      }else{
          out_shape = get_conv_output_size(input_dims, ksize, stride, padding, dilation);
      }
  }else{
      out_shape = input_dims;
  }
  for (auto& v : out_shape){
      if (v <= 0){
          TV_THROW_RT_ERR("your out spatial shape", out_shape, "ratch zero!, input shape:", input_dims);
      }
  }
  std::vector<int64_t> output_dims_i64(out_shape.begin(), out_shape.end());
  int64_t out_spatial_volume = std::accumulate(output_dims_i64.begin(),
    output_dims_i64.end(), int64_t(1), std::multiplies<int64_t>()) * batch_size;
  bool use_int64_hash_k = out_spatial_volume >= int64_t(std::numeric_limits<int>::max());
  tv::DType indice_uniq_dtype = use_int64_hash_k ? tv::int64 : tv::int32;
  tv::Tensor pair;
  int64_t num_act_in_bounded = indices.dim(0);
  if (num_out_act_bound > 0){
      // we need stable pair stride for bounded output
      num_act_in_bounded = num_input_act_bound;
  }
  pair = allocator.full_int("PairFwd", 
      {2, kv, num_act_in_bounded}, -1, indices.dtype(), indices.device(), stream_int);
  auto indice_num_per_loc = allocator.zeros("IndiceNumPerLoc", 
      {kv}, indices.dtype(), indices.device(), stream_int);
  tv::Tensor out_inds;
  int num_act_out = -1;
  if (subm){
    num_act_out = indices.dim(0);
    if (indices.is_cpu()){
        generate_subm_conv_inds_cpu(indices, pair, out_inds, indice_num_per_loc,
            batch_size, input_dims, ksize, dilation);
    }
    else {
        ExternalAllocator::guard_t hash_k_guard, hash_v_gurad, hash_kv_gurad;
        out_inds = indices;
        int num_points = out_inds.dim(0);
        tv::Tensor hash_k, hash_v;
        if (use_int64_hash_k){
            hash_k_guard = allocator.empty_guard({num_points * 2}, 
                tv::int64, 0, "HashKOrKV");
            hash_v_gurad = allocator.empty_guard({num_points * 2}, 
                tv::int32, 0, "HashV");
            hash_k = hash_k_guard->tensor;
            hash_v = hash_v_gurad->tensor;
        }else{
            hash_kv_gurad = allocator.empty_guard({2, num_points * 2}, 
                tv::int32, 0, "HashKOrKV");
            hash_k = hash_kv_gurad->tensor[0];
            hash_v = hash_kv_gurad->tensor[1];
        }
        generate_subm_conv_inds(indices, hash_k, hash_v, pair, out_inds, indice_num_per_loc,
            batch_size, input_dims, ksize, dilation, tv::Tensor(), false, stream_int);
    }
  }
  else{
    if (indices.is_cpu()){
        TV_ASSERT_RT_ERR(num_out_act_bound <= 0, "cpu algo don't support out bound")
        out_inds = allocator.empty("OutIndices", 
            {kv * indices.dim(0), indices.dim(1)}, indices.dtype(), -1);
        num_act_out = generate_conv_inds_cpu(indices, pair, out_inds, indice_num_per_loc,
            batch_size, out_shape, input_dims, ksize, 
            stride, padding, dilation, transposed);
    }
                    else {
                        ThrustAllocator thrustalloc(allocator);
                        auto tvctx = tv::Context();
                        tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream_int));
                        auto indice_pairs_uniq_guard = allocator.empty_guard(
                            {int64_t(pair.numel() / 2 + 1)}, indice_uniq_dtype, 0, 
                            "IndicePairsUniq");
                        auto indice_pairs_uniq = indice_pairs_uniq_guard->tensor;
                        auto indice_pairs_uniq_bkp_guard = allocator.empty_guard(
                            {int64_t(pair.numel() / 2 + 1)}, indice_uniq_dtype, 0,
                            "IndicePairsUniqBackup");
                        generate_conv_inds_stage1(indices, pair, indice_pairs_uniq,
                            indice_num_per_loc, batch_size, out_shape, input_dims, ksize,
                            stride, padding, dilation, transposed, stream_int);
                        indice_pairs_uniq_bkp_guard->tensor.copy_(indice_pairs_uniq, tvctx);
                        // TODO pytorch unique may be faster?
                        num_act_out = apply_thrust_unique_to_indice_pairs_uniq(indice_pairs_uniq, thrustalloc, stream_int);
                        if (num_act_out == 0){
                            std::stringstream ss;
                            ss << R"(Your points vanished here, this usually because you provide 
    conv params that may ignore some input points. Example: 
        spatial_shape=[8, 200, 200]
        ksize=3
        stride=2
        padding=[0, 1, 1]
        dilation=1
        Coordinates=[[0, 7, 153, 142]]
    these params will cause ALL points in z == 7 dropped because of padding_z=0.
    enlarge your spatial shape or change your conv param to make sure 
    every input point has a corresponding output point.
    Your Conv Params: )" << "\n";
                            tv::sstream_print<'\0'>(ss, "    spatial_shape=", input_dims, "\n");
                            tv::sstream_print<'\0'>(ss, "    ksize=", ksize, "\n");
                            tv::sstream_print<'\0'>(ss, "    stride=", stride, "\n");
                            tv::sstream_print<'\0'>(ss, "    padding=", padding, "\n");
                            tv::sstream_print<'\0'>(ss, "    dilation=", dilation, "\n");
                            tv::ssprint(ss.str());
                            throw std::runtime_error(ss.str());
                        }
                        bool use_bound_algo = false;
                        int64_t num_out_bounded = num_act_out;
                        if (num_out_act_bound > 0 && num_act_out > num_out_act_bound){
                            num_act_out = num_out_act_bound;
                            use_bound_algo = true;
                        }
                        if (num_out_act_bound > 0 ){
                            num_out_bounded = num_out_act_bound;
                        }
                        indice_pairs_uniq = indice_pairs_uniq.slice_first_axis(0, num_act_out);
                        out_inds = allocator.empty("OutIndices", 
                            {num_out_bounded, indices.dim(1)}, indices.dtype(), 0, stream_int);
                        ExternalAllocator::guard_t hash_k_guard, hash_v_gurad, hash_kv_gurad;
                        tv::Tensor hash_k, hash_v;
                        if (use_int64_hash_k){
                            hash_k_guard = allocator.empty_guard({num_act_out * 2}, 
                                tv::int64, 0, "HashKOrKV");
                            hash_v_gurad = allocator.empty_guard({num_act_out * 2}, 
                                tv::int32, 0, "HashV");
                            hash_k = hash_k_guard->tensor;
                            hash_v = hash_v_gurad->tensor;
                        }else{
                            hash_kv_gurad = allocator.empty_guard({2, num_act_out * 2}, 
                                tv::int32, 0, "HashKOrKV");
                            hash_k = hash_kv_gurad->tensor[0];
                            hash_v = hash_kv_gurad->tensor[1];
                        }
                        num_act_out = generate_conv_inds_stage2(indices, hash_k, hash_v, pair,
                            indice_pairs_uniq, indice_pairs_uniq_bkp_guard->tensor, 
                            out_inds, indice_num_per_loc, num_act_out,
                            batch_size, out_shape, input_dims, ksize, stride, padding, dilation,
                            transposed, stream_int, use_bound_algo);
                    }
  }
  return num_act_out;
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib