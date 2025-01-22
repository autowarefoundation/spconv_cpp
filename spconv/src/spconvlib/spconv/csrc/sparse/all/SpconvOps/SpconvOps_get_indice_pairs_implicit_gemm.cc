#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/all/HashCoreHost.h>
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
using HashCoreHost = spconvlib::spconv::csrc::sparse::all::HashCoreHost;
std::tuple<tv::Tensor, int> SpconvOps::get_indice_pairs_implicit_gemm(ExternalAllocator& allocator, tv::Tensor indices, int batch_size, std::vector<int> input_dims, int algo, std::vector<int> ksize, std::vector<int> stride, std::vector<int> padding, std::vector<int> dilation, std::vector<int> out_padding, bool subm, bool transposed, bool is_train, std::uintptr_t stream_int, int num_out_act_bound, tv::CUDAKernelTimer timer, bool direct_table, bool do_sort, std::unordered_map<std::string, tv::Tensor> preallocated)   {
  
  auto tvctx = tv::Context();
  tvctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(stream_int));
  auto conv_algo = static_cast<tv::gemm::SparseConvAlgo>(algo);
  int kv = std::accumulate(ksize.begin(), ksize.end(), 1, std::multiplies<int>());
  int mask_int_count = tv::div_up(kv, 32);
  // if (mask_int_count > 1 && mask_int_count < 4)
  //     mask_int_count = 4;
  // TV_ASSERT_RT_ERR(mask_int_count == 1 || mask_int_count == 4, "Not Implement too large kernel");
  // TV_ASSERT_RT_ERR(kv <= 32, "currently only support ksize < 32");
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
  TV_ASSERT_RT_ERR(conv_algo == tv::gemm::SparseConvAlgo::kMaskImplicitGemm || 
      conv_algo == tv::gemm::SparseConvAlgo::kMaskSplitImplicitGemm, "only support implicit gemm");
  bool is_mask_split = conv_algo == tv::gemm::SparseConvAlgo::kMaskSplitImplicitGemm;
  int mask_split_count = is_mask_split ? 2 : 1;
  int64_t num_act_in = indices.dim(0);
  tv::Tensor pair;
  if (subm){
      if (preallocated.find("PairFwd") != preallocated.end()){
          pair = preallocated.at("PairFwd");
      }
      else{
          if (is_train){
              // query pair for fwd and bwd
              pair = allocator.full_int("PairFwd", 
                  {2, kv, num_act_in}, -1, indices.dtype(), indices.device(), stream_int);
          }else{
              // query pair fwd only
              pair = allocator.full_int("PairFwd", 
                  {1, kv, num_act_in}, -1, indices.dtype(), indices.device(), stream_int);
          }
      }
  }else{
      if (is_train){
          // query pair bwd
          pair = allocator.full_int("PairBwd", 
              {kv, num_act_in}, -1, indices.dtype(), indices.device(), stream_int);
      }else{
          // don't need pair bwd, empty
          pair = tv::Tensor();
      }
  }
  tv::Tensor indice_num_per_loc;
  if (preallocated.find("IndiceNumPerLoc") != preallocated.end()){
      indice_num_per_loc = preallocated.at("IndiceNumPerLoc");
  }
  else{
      indice_num_per_loc = allocator.zeros("IndiceNumPerLoc", 
       {kv}, indices.dtype(), indices.device(), stream_int);
  }
  tv::Tensor mask_tensor = tv::zeros({mask_split_count}, tv::uint32, -1);
  auto mask_tensor_ptr = mask_tensor.data_ptr<uint32_t>();
  if (is_mask_split){
      TV_ASSERT_RT_ERR(mask_int_count == 1, "not support for kv > 32");
      auto kv_div_2 = kv / 2;
      auto remain = kv - kv_div_2;
      uint64_t mask_np_1 = 1;
      uint64_t first = ((mask_np_1 << remain) - 1);
      uint64_t second = ((mask_np_1 << kv_div_2) - 1) << remain;
      mask_tensor_ptr[0] = uint32_t(first);
      mask_tensor_ptr[1] = uint32_t(second);
  }
  else{
      mask_tensor_ptr[0] = 0xffffffff;
  }
  tv::Tensor out_inds;
  ThrustAllocator thrustalloc(allocator);
  int num_act_out = 0;
  if (subm){
    ExternalAllocator::guard_t hash_k_guard, hash_v_gurad, hash_kv_gurad;
    out_inds = indices;
    num_act_out = indices.dim(0);
    int hash_size = out_inds.dim(0) * 2;
    tv::Tensor hash_k, hash_v;
    if (use_int64_hash_k){
        hash_k_guard = allocator.empty_guard({hash_size}, 
            tv::int64, 0, "HashKOrKV");
        hash_v_gurad = allocator.empty_guard({hash_size}, 
            tv::int32, 0, "HashV");
        hash_k = hash_k_guard->tensor;
        hash_v = hash_v_gurad->tensor;
    }else{
        if (preallocated.find("HashKOrKV") != preallocated.end()){
            auto hash_kv = preallocated.at("HashKOrKV");
            hash_k = hash_kv[0];
            hash_v = hash_kv[1];
        }else{
            hash_kv_gurad = allocator.empty_guard({2, hash_size}, 
                tv::int32, 0, "HashKOrKV");
            hash_k = hash_kv_gurad->tensor[0];
            hash_v = hash_kv_gurad->tensor[1];
        }
    }
    tv::Tensor pair_mask;
    if (preallocated.find("PairMask") != preallocated.end()){
        pair_mask = preallocated.at("PairMask");
    }else{
        pair_mask = allocator.empty("PairMask", 
            {mask_split_count, num_act_in, mask_int_count}, tv::uint32, 0, stream_int);
    }
    generate_subm_conv_inds(indices, hash_k, hash_v, pair, out_inds, indice_num_per_loc,
        batch_size, input_dims, ksize, dilation, pair_mask, is_train, stream_int);
    auto mask_argsort = allocator.empty("MaskArgSort", 
        {mask_split_count, num_act_in}, tv::int32, 0, stream_int);
    for (int j = 0; j < mask_split_count; ++j){
        sort_1d_by_key_allocator_v2(pair_mask[j], thrustalloc, mask_argsort[j], stream_int, mask_int_count, do_sort);
    }
  }
  else{
                // auto start = tv::CPUEvent().record(stream_int);
                auto pair_bwd = pair;
                auto pair_size = kv * num_act_in;
                ExternalAllocator::guard_t hash_k_guard, hash_v_gurad, hash_kv_gurad;
                ExternalAllocator::guard_t indice_pairs_uniq_guard, indice_pairs_uniq_bkp_guard;
                tv::Tensor hash_k, hash_v, indice_pairs_uniq;
                int max_num_act = get_handcrafted_max_act_out(num_act_in, ksize, stride, padding, dilation);
                if (transposed){
                    max_num_act = pair_size;
                }
                int hash_size = int(max_num_act * 1.1);
                if (direct_table){
                    if (use_int64_hash_k){
                        // temp memory don't need to be fixed, static alloc will check
                        // that tensor is large enough.
                        hash_k_guard = allocator.empty_guard({hash_size}, 
                            tv::int64, 0, "HashKOrKV");
                        hash_v_gurad = allocator.empty_guard({hash_size}, 
                            tv::int32, 0, "HashV");
                        hash_k = hash_k_guard->tensor;
                        hash_v = hash_v_gurad->tensor;
                    }else{
                        hash_kv_gurad = allocator.empty_guard({2, hash_size}, 
                            tv::int32, 0, "HashKOrKV");
                        hash_k = hash_kv_gurad->tensor[0];
                        hash_v = hash_kv_gurad->tensor[1];
                    }
                }
                indice_pairs_uniq_guard = allocator.empty_guard({int64_t(pair_size + 1)}, 
                    indice_uniq_dtype, 0, "IndicePairsUniq");
                indice_pairs_uniq = indice_pairs_uniq_guard->tensor;
                // auto indice_pairs_uniq_bkp = indice_pairs_uniq_guard->tensor[1];
                indice_pairs_uniq_bkp_guard = allocator.empty_guard({int64_t(pair_size + 1)}, 
                    indice_uniq_dtype, 0, "IndicePairsUniqBackup");
                auto indice_pairs_uniq_bkp = indice_pairs_uniq_bkp_guard->tensor;
                {
                    tv::CUDAKernelTimerGuard timer_guard("gen_conv_inds_stage1", 
                        timer, reinterpret_cast<cudaStream_t>(stream_int));
                    if (direct_table){
                        generate_conv_inds_mask_stage1_direct_table(indices, 
                        hash_k, hash_v, pair_bwd, indice_pairs_uniq_bkp,
                        indice_num_per_loc, batch_size, out_shape, input_dims, ksize,
                        stride, padding, dilation, transposed, stream_int);
                    }else{
                        generate_conv_inds_mask_stage1(indices, pair_bwd, indice_pairs_uniq,
                            indice_num_per_loc, batch_size, out_shape, input_dims, ksize,
                            stride, padding, dilation, transposed, stream_int);
                        indice_pairs_uniq_bkp.copy_(indice_pairs_uniq, tvctx);
                    }
                }
                // TODO pytorch unique run faster.
                {
                    tv::CUDAKernelTimerGuard timer_guard(std::string("unique_") + std::to_string(indice_pairs_uniq.dim(0)), 
                        timer, reinterpret_cast<cudaStream_t>(stream_int));
                    if (direct_table){
                        auto uniqcnt = allocator.zeros_guard({1}, tv::int32, 0, 
                            "TightUniqueCount", stream_int);
                        num_act_out = unique_hash(hash_k, hash_v, uniqcnt->tensor, 
                            indice_pairs_uniq, num_out_act_bound, stream_int);
                    }else{
                        num_act_out = apply_thrust_unique_to_indice_pairs_uniq(indice_pairs_uniq, thrustalloc, stream_int);
                    }
                }
                // tv::ssprint("HASH SIZE", hash_size, num_act_out);
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
                if (num_out_act_bound > 0 && num_act_out > num_out_act_bound){
                    num_act_out = num_out_act_bound;
                }
                indice_pairs_uniq = indice_pairs_uniq.slice_first_axis(0, num_act_out);
                // for fixed size allocator, all memory alloc size must be fixed.
                tv::Tensor pair_fwd, pair_mask_fwd, pair_mask_bwd;
                {
                    tv::CUDAKernelTimerGuard timer_guard("alloc_stage2", 
                        timer, reinterpret_cast<cudaStream_t>(stream_int));
                    out_inds = allocator.empty("OutIndices", 
                        {num_act_out, indices.dim(1)}, indices.dtype(), 0, stream_int);
                    pair_fwd = allocator.full_int("PairFwd", 
                        {kv, num_act_out}, -1, indices.dtype(), indices.device(), stream_int);
                    pair_mask_fwd = allocator.zeros("PairMask", 
                        {mask_split_count, num_act_out, mask_int_count}, tv::uint32, 0, stream_int);
                    pair_mask_bwd = tv::Tensor();
                    if (is_train){
                        pair_mask_bwd = allocator.zeros("PairMaskBwd", 
                            {mask_split_count, indices.dim(0), mask_int_count}, tv::uint32, 0, stream_int);
                    }
                }
                if (!direct_table){
                    int hash_size = int(num_act_out * 2);
                    if (use_int64_hash_k){
                        // temp memory don't need to be fixed, static alloc will check
                        // that tensor is large enough.
                        hash_k_guard = allocator.empty_guard({hash_size}, 
                            tv::int64, 0, "HashKOrKV");
                        hash_v_gurad = allocator.empty_guard({hash_size}, 
                            tv::int32, 0, "HashV");
                        hash_k = hash_k_guard->tensor;
                        hash_v = hash_v_gurad->tensor;
                    }else{
                        hash_kv_gurad = allocator.empty_guard({2, hash_size}, 
                            tv::int32, 0, "HashKOrKV");
                        hash_k = hash_kv_gurad->tensor[0];
                        hash_v = hash_kv_gurad->tensor[1];
                    }
                }
                {
                    tv::CUDAKernelTimerGuard timer_guard(std::string("gen_conv_inds_stage2_") + std::to_string(num_act_out), 
                        timer, reinterpret_cast<cudaStream_t>(stream_int));
                    if (direct_table){
                        assign_output_direct_hash(indice_pairs_uniq, out_inds, 
                            batch_size, out_shape, 
                            input_dims, ksize, stride, padding, dilation, stream_int);
                        generate_conv_inds_stage2_mask_direct_table(indices, hash_k, hash_v, pair_fwd, pair_bwd,
                            indice_pairs_uniq, indice_pairs_uniq_bkp, 
                            out_inds, pair_mask_fwd, pair_mask_bwd, num_act_out,
                            batch_size, out_shape, input_dims, ksize, stride, padding, dilation,
                            transposed, stream_int);
                    }else{
                        generate_conv_inds_mask_stage2(indices, hash_k, hash_v, pair_fwd, pair_bwd,
                            indice_pairs_uniq, indice_pairs_uniq_bkp, 
                            out_inds, pair_mask_fwd, pair_mask_bwd, num_act_out,
                            batch_size, out_shape, input_dims, ksize, stride, padding, dilation,
                            transposed, stream_int);
                    }
                }
    auto mask_argsort_fwd = allocator.empty("MaskArgSort", 
        {mask_split_count, num_act_out}, tv::int32, 0, stream_int);
    tv::Tensor mask_argsort_bwd = tv::Tensor();
    if (is_train){
        mask_argsort_bwd = allocator.zeros("MaskArgSortBwd", 
            {mask_split_count, num_act_in}, tv::int32, 0, stream_int);
    }
    {
        tv::CUDAKernelTimerGuard timer_guard("gen_conv_inds_sort", 
            timer, reinterpret_cast<cudaStream_t>(stream_int));
        if (is_mask_split){
            TV_ASSERT_RT_ERR(do_sort, "not implemented for now");
            for (int j = 0; j < mask_split_count; ++j){
                auto mask_tensor_sub = mask_tensor.slice_first_axis(j, j + 1);
                if (!is_train){
                    sort_1d_by_key_split_allocator_v2(pair_mask_fwd[j], thrustalloc, 
                        mask_tensor_sub, mask_argsort_fwd[j], stream_int);
                }else{
                    sort_1d_by_key_split_allocator_v2(pair_mask_fwd[j], thrustalloc, 
                        mask_tensor_sub, mask_argsort_fwd[j], stream_int);
                    sort_1d_by_key_split_allocator_v2(pair_mask_bwd[j], thrustalloc, 
                        mask_tensor_sub, mask_argsort_bwd[j], stream_int);
                }
            }
        }else{
            if (!is_train){
                sort_1d_by_key_allocator_v2(pair_mask_fwd[0], thrustalloc, 
                    mask_argsort_fwd[0], stream_int, mask_int_count, do_sort);
            }else{
                sort_1d_by_key_allocator_v2(pair_mask_fwd[0], thrustalloc, 
                    mask_argsort_fwd[0], stream_int, mask_int_count, do_sort);
                sort_1d_by_key_allocator_v2(pair_mask_bwd[0], thrustalloc, 
                    mask_argsort_bwd[0], stream_int, mask_int_count, do_sort);
            }
        }
    }
  }
  return std::make_tuple(mask_tensor, num_act_out);
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib