#include <spconvlib/spconv/csrc/sparse/all/ops_cpu4d/SparseConvIndicesCPU.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops_cpu4d {
using TensorView = spconvlib::cumm::common::TensorView;
using ConvLocIter = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::spinds::ConvOutLocIter;
using ConvProblem = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::spinds::ConvProblem;
using ConvLocIter64 = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::spinds64::ConvOutLocIter;
int SparseConvIndicesCPU::generate_subm_conv_inds(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 4> input_dims, tv::array<int, 4> ksize, tv::array<int, 4> dilation)   {
  
  tv::array<int, 4> stride, padding;
  for (int i = 0; i < 4; ++i){
      TV_ASSERT_RT_ERR(ksize[i] % 2 == 1, "subm only support odd ksize");
      stride[i] = 1;
      padding[i] = (ksize[i] / 2) * dilation[i];
  }
  int kv = ksize.op<tv::arrayops::prod>();
  TV_ASSERT_RT_ERR(input_dims.op<tv::arrayops::prod>() < std::numeric_limits<int32_t>::max(), 
      "kernel volume must smaller than max value of int32_t");
  ConvProblem problem(batch_size, 1, 1, input_dims, input_dims, ksize, padding, stride, dilation);
  bool use_int32 = problem.check_npq_not_overflow();
  if (int(use_int32) == 0){
    ConvLocIter64 loc_iter(problem);
    int indices_pair_size = indice_pairs.dim(2);
    int indices_pair_size_mul_RS = indices_pair_size * kv;
    auto indice_pairs_ptr = indice_pairs.data_ptr<int32_t>();
    std::unordered_map<int32_t, int32_t> hash;
    auto indices_ptr = indices.data_ptr<const int32_t>();
    int indice_in_num = indices.dim(0);
    for (int i = 0; i < indice_in_num; ++i){
        int32_t index = loc_iter.layout_npq(indices_ptr);
        hash.insert({index, i});
        indices_ptr += 5;
    }
    for (int filter_offset = 0; filter_offset < (kv / 2 + 1); ++filter_offset){
        int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
        int filter_offset_mul_indices_pair_size_1 = (kv - 1 - filter_offset) * indices_pair_size;
        if (filter_offset == kv / 2){
            for (int i = 0; i < indice_in_num; ++i){
                indice_pairs_ptr[filter_offset_mul_indices_pair_size + i] = i;
                indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + i] = i;
            }
        }else{
            indices_ptr = indices.data_ptr<const int32_t>();
            auto indice_num_per_loc_ptr = indice_num_per_loc.data_ptr<int32_t>() + filter_offset;
            for (int i = 0; i < indice_in_num; ++i){
                tv::array<int, 5> npq_offset;
                if (loc_iter.query_npq_no_stride(indices_ptr, npq_offset)){
                    auto index = loc_iter.layout_npq(npq_offset);
                    auto iter = hash.find(index);
                    if (iter != hash.end()){
                        auto old_num = indice_num_per_loc_ptr[0]++;
                        indice_pairs_ptr[filter_offset_mul_indices_pair_size + old_num] = i;
                        indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + old_num] = iter->second;
                        indice_pairs_ptr[filter_offset_mul_indices_pair_size_1 + old_num] = iter->second;
                        indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size_1 + old_num] = i;
                    }
                }
                indices_ptr += 5;
            }
        }
        ++loc_iter;
    }
  }
  else if (int(use_int32) == 1){
    ConvLocIter loc_iter(problem);
    int indices_pair_size = indice_pairs.dim(2);
    int indices_pair_size_mul_RS = indices_pair_size * kv;
    auto indice_pairs_ptr = indice_pairs.data_ptr<int32_t>();
    std::unordered_map<int32_t, int32_t> hash;
    auto indices_ptr = indices.data_ptr<const int32_t>();
    int indice_in_num = indices.dim(0);
    for (int i = 0; i < indice_in_num; ++i){
        int32_t index = loc_iter.layout_npq(indices_ptr);
        hash.insert({index, i});
        indices_ptr += 5;
    }
    for (int filter_offset = 0; filter_offset < (kv / 2 + 1); ++filter_offset){
        int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
        int filter_offset_mul_indices_pair_size_1 = (kv - 1 - filter_offset) * indices_pair_size;
        if (filter_offset == kv / 2){
            for (int i = 0; i < indice_in_num; ++i){
                indice_pairs_ptr[filter_offset_mul_indices_pair_size + i] = i;
                indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + i] = i;
            }
        }else{
            indices_ptr = indices.data_ptr<const int32_t>();
            auto indice_num_per_loc_ptr = indice_num_per_loc.data_ptr<int32_t>() + filter_offset;
            for (int i = 0; i < indice_in_num; ++i){
                tv::array<int, 5> npq_offset;
                if (loc_iter.query_npq_no_stride(indices_ptr, npq_offset)){
                    auto index = loc_iter.layout_npq(npq_offset);
                    auto iter = hash.find(index);
                    if (iter != hash.end()){
                        auto old_num = indice_num_per_loc_ptr[0]++;
                        indice_pairs_ptr[filter_offset_mul_indices_pair_size + old_num] = i;
                        indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + old_num] = iter->second;
                        indice_pairs_ptr[filter_offset_mul_indices_pair_size_1 + old_num] = iter->second;
                        indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size_1 + old_num] = i;
                    }
                }
                indices_ptr += 5;
            }
        }
        ++loc_iter;
    }
  }
  else{
    TV_THROW_RT_ERR("unknown val int(use_int32), available: [0, 1]")
  }
  return indices.dim(0);
}
} // namespace ops_cpu4d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib