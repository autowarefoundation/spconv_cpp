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
int SparseConvIndicesCPU::generate_conv_inds(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 4> output_dims, tv::array<int, 4> input_dims, tv::array<int, 4> ksize, tv::array<int, 4> stride, tv::array<int, 4> padding, tv::array<int, 4> dilation, bool transposed)   {
  
  int kv = ksize.op<tv::arrayops::prod>();
  ConvProblem problem(batch_size, 1, 1, input_dims, output_dims, ksize, padding, stride, dilation);
  bool use_int32 = problem.check_npq_not_overflow();
  int num_act = 0;
  if (int(use_int32) == 0){
    ConvLocIter64 loc_iter(problem);
    int indices_pair_size = indice_pairs.dim(2);
    int indices_pair_size_mul_RS = indices_pair_size * kv;
    auto indice_pairs_ptr = indice_pairs.data_ptr<int32_t>();
    std::unordered_map<int32_t, int32_t> hash;
    auto indices_ptr = indices.data_ptr<const int32_t>();
    auto out_inds_ptr = out_inds.data_ptr<int32_t>();
    TV_ASSERT_RT_ERR(input_dims.op<tv::arrayops::prod>() < std::numeric_limits<int32_t>::max(), 
        "kernel volume must smaller than max value of int32_t");
    int indice_in_num = indices.dim(0);
    int32_t hashval;
    for (int filter_offset = 0; filter_offset < kv; ++filter_offset){
        int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
        indices_ptr = indices.data_ptr<const int32_t>();
        auto indice_num_per_loc_ptr = indice_num_per_loc.data_ptr<int32_t>() + filter_offset;
        for (int i = 0; i < indice_in_num; ++i){
            tv::array<int, 5> npq_offset;
            bool valid;
            if (transposed){
                valid = loc_iter.query_nhw_out(indices_ptr, npq_offset);
            }else{
                valid = loc_iter.query_npq(indices_ptr, npq_offset);
            }
            if (valid){
                auto index = loc_iter.layout_npq(npq_offset);
                auto iter = hash.find(index);
                if (iter == hash.end()){
                    hashval = num_act++;
                    hash.insert({index, hashval});
                    for (int k = 0; k < 5; ++k){
                        out_inds_ptr[k] = npq_offset[k];
                    }
                    out_inds_ptr += 5;
                }else{
                    hashval = iter->second;
                }
                indice_pairs_ptr[filter_offset_mul_indices_pair_size + indice_num_per_loc_ptr[0]] = i;
                indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + indice_num_per_loc_ptr[0]++] = hashval;
            }
            indices_ptr += 5;
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
    auto out_inds_ptr = out_inds.data_ptr<int32_t>();
    TV_ASSERT_RT_ERR(input_dims.op<tv::arrayops::prod>() < std::numeric_limits<int32_t>::max(), 
        "kernel volume must smaller than max value of int32_t");
    int indice_in_num = indices.dim(0);
    int32_t hashval;
    for (int filter_offset = 0; filter_offset < kv; ++filter_offset){
        int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
        indices_ptr = indices.data_ptr<const int32_t>();
        auto indice_num_per_loc_ptr = indice_num_per_loc.data_ptr<int32_t>() + filter_offset;
        for (int i = 0; i < indice_in_num; ++i){
            tv::array<int, 5> npq_offset;
            bool valid;
            if (transposed){
                valid = loc_iter.query_nhw_out(indices_ptr, npq_offset);
            }else{
                valid = loc_iter.query_npq(indices_ptr, npq_offset);
            }
            if (valid){
                auto index = loc_iter.layout_npq(npq_offset);
                auto iter = hash.find(index);
                if (iter == hash.end()){
                    hashval = num_act++;
                    hash.insert({index, hashval});
                    for (int k = 0; k < 5; ++k){
                        out_inds_ptr[k] = npq_offset[k];
                    }
                    out_inds_ptr += 5;
                }else{
                    hashval = iter->second;
                }
                indice_pairs_ptr[filter_offset_mul_indices_pair_size + indice_num_per_loc_ptr[0]] = i;
                indice_pairs_ptr[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + indice_num_per_loc_ptr[0]++] = hashval;
            }
            indices_ptr += 5;
        }
        ++loc_iter;
    }
  }
  else{
    TV_THROW_RT_ERR("unknown val int(use_int32), available: [0, 1]")
  }
  return num_act;
}
} // namespace ops_cpu4d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib