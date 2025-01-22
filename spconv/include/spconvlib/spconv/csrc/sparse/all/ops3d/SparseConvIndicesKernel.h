#pragma once
#include <tensorview/hash/ops.h>
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/cumm/common/TensorViewKernel.h>
#include <spconvlib/cumm/common/TensorViewHashKernel.h>
#include <spconvlib/spconv/csrc/sparse/all/ops3d/spinds/ConvOutLocIter.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu3d/spinds/ConvProblem.h>
#include <spconvlib/spconv/csrc/sparse/all/ops3d/spinds64/ConvOutLocIter.h>
#include <spconvlib/spconv/csrc/sparse/all/ops3d/cudakers/CudaCommonKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops3d {
using TensorView = spconvlib::cumm::common::TensorView;
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
using TensorViewHashKernel = spconvlib::cumm::common::TensorViewHashKernel;
using ConvLocIter = spconvlib::spconv::csrc::sparse::all::ops3d::spinds::ConvOutLocIter;
using ConvProblem = spconvlib::spconv::csrc::sparse::all::ops_cpu3d::spinds::ConvProblem;
using ConvLocIter64 = spconvlib::spconv::csrc::sparse::all::ops3d::spinds64::ConvOutLocIter;
template <typename TIndiceUniq, typename TConvLocIter>
__global__ void calc_conv_indices_stage1(TConvLocIter loc_iter, const int* indices_in, int32_t* indice_pairs, TIndiceUniq* indice_pairs_for_uniq, int* indice_num_per_loc, int num_indices_in, int indices_pair_size, int RS, bool transposed)   {
  
  int filter_offset = blockIdx.y;
  loc_iter.set_filter_offset(filter_offset);
  // int indices_pair_size_mul_RS = indices_pair_size * RS;
  int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
  for (int i : tv::KernelLoopX<int>(num_indices_in)) {
      tv::array<int, 4> npq_offset;
      bool valid;
      if (transposed){
          valid = loc_iter.query_nhw_out(indices_in + i * 4, npq_offset);
      }else{
          valid = loc_iter.query_npq(indices_in + i * 4, npq_offset);
      }
      if (valid){
          int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
          int64_t offset = loc_iter.layout_npq(npq_offset);
          if (old_num < indices_pair_size){
              indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;
              // indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + old_num] = offset;
              indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + old_num] = offset;
          }
      }
  }
}
template <typename TTable, typename TLayoutNPQ>
__global__ void build_conv_hash_table(TTable table, int* indices_out, const typename TTable::key_type* indice_pairs_for_uniq, TLayoutNPQ layout_npq, int num_indices)   {
  
  for (int output_index : tv::KernelLoopX<int>(num_indices)) {
      auto output_coord_offset = indice_pairs_for_uniq[output_index];
      layout_npq.inverse(output_coord_offset, indices_out + 4 * output_index);
      table.insert(output_coord_offset, output_index);
  }
}
template <typename TTable, typename TLayoutNPQ>
__global__ void arange_hash_table_and_assign_out(TTable table, int* indices_out, int* count, int limit, TLayoutNPQ layout_npq)   {
  
  auto key_ptr = table.key_ptr();
  auto value_ptr = table.value_ptr();
  for (auto i : tv::KernelLoopX<int>(table.size())) {
      auto output_coord_offset = key_ptr[i];
      if (output_coord_offset != TTable::empty_key) {
          auto output_index = tv::cuda::atomicAggInc(count);
          if (output_index < limit){
              value_ptr[i] = output_index;
              layout_npq.inverse(output_coord_offset, indices_out + 4 * output_index);
          }else{
              value_ptr[i] = -1;
          }
      }
  }
}
template <typename TTable>
__global__ void arange_hash_table(TTable table, typename TTable::key_type * out_indices_offset, int* count, int limit)   {
  
  auto key_ptr = table.key_ptr();
  auto value_ptr = table.value_ptr();
  for (auto i : tv::KernelLoopX<int>(table.size())) {
      auto output_coord_offset = key_ptr[i];
      if (output_coord_offset != TTable::empty_key) {
          auto output_index = tv::cuda::atomicAggInc(count);
          value_ptr[i] = output_index < limit ? output_index : -1;
          out_indices_offset[output_index] = output_coord_offset;
      }
  }
}
template <typename T, typename TLayoutNPQ>
__global__ void assign_out_indices(int* indices_out, const T* out_indices_offset, TLayoutNPQ layout_npq, int size)   {
  
  for (auto i : tv::KernelLoopX<int>(size)) {
      layout_npq.inverse(out_indices_offset[i], indices_out + 4 * i);
  }
}
template <typename TTable>
__global__ void calc_conv_indices_stage2(TTable table, const typename TTable::key_type* indice_pairs_uniq_before_sort, int* indice_pairs_out_part, int num_indices_in, int indices_pair_size)   {
  
  int filter_offset = blockIdx.y;
  auto indice_pairs_out_part_filter = indice_pairs_out_part + filter_offset * indices_pair_size;
  auto indice_pairs_uniq_before_sort_filter = indice_pairs_uniq_before_sort + filter_offset * indices_pair_size;
  for (int i : tv::KernelLoopX<int>(num_indices_in)) {
      int32_t output_coord_offset = indice_pairs_uniq_before_sort_filter[i];
      if (output_coord_offset != std::numeric_limits<typename TTable::key_type>::max()){
          auto table_offset = table.lookup_offset(output_coord_offset);
          if (table_offset != -1){
              indice_pairs_out_part_filter[i] = table.value_ptr()[table_offset];
          }
      }
  }
}
template <typename TTable>
__global__ void calc_conv_indices_stage2_bounded(TTable table, const typename TTable::key_type* indice_pairs_uniq_before_sort, const int* indice_pairs_in_part_temp, int* indice_pairs_in_part, int* indice_pairs_out_part, int* indice_num_per_loc, int num_indices_in, int indices_pair_size)   {
  
  int filter_offset = blockIdx.y;
  auto indice_pairs_in_part_filter = indice_pairs_in_part + filter_offset * indices_pair_size;
  auto indice_pairs_out_part_filter = indice_pairs_out_part + filter_offset * indices_pair_size;
  auto indice_pairs_in_part_temp_filter = indice_pairs_in_part_temp + filter_offset * indices_pair_size;
  auto indice_pairs_uniq_before_sort_filter = indice_pairs_uniq_before_sort + filter_offset * indices_pair_size;
  for (int i : tv::KernelLoopX<int>(num_indices_in)) {
      int32_t output_coord_offset = indice_pairs_uniq_before_sort_filter[i];
      if (output_coord_offset != std::numeric_limits<typename TTable::key_type>::max()){
          auto table_offset = table.lookup_offset(output_coord_offset);
          if (table_offset != -1){
              int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
              indice_pairs_in_part_filter[old_num] = indice_pairs_in_part_temp_filter[i];
              indice_pairs_out_part_filter[old_num] = table.value_ptr()[table_offset];
          }
      }
  }
}
template <typename TIndiceUniq, typename TConvLocIter>
__global__ void calc_conv_indices_stage1_mask(TConvLocIter loc_iter, const int* indices_in, int32_t* indice_pairs_bwd, TIndiceUniq* indice_pairs_for_uniq, int* indice_num_per_loc, int num_indices_in, int RS, bool transposed)   {
  
  int filter_offset = blockIdx.y;
  loc_iter.set_filter_offset(filter_offset);
  // int indices_pair_size_mul_RS = num_indices_in * RS;
  int filter_offset_mul_indices_pair_size = filter_offset * num_indices_in;
  for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {
      tv::array<int, 4> npq_offset;
      bool valid;
      if (transposed){
          valid = loc_iter.query_nhw_out(indices_in + input_index * 4, npq_offset);
      }else{
          valid = loc_iter.query_npq(indices_in + input_index * 4, npq_offset);
      }
      if (valid){
          // int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
          TIndiceUniq output_coord_offset = loc_iter.layout_npq(npq_offset);
          // if (old_num < indices_pair_size){
          // indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;
          // indice_pairs_bwd[filter_offset_mul_indices_pair_size + input_index] = output_coord_offset;
          // indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + old_num] = output_coord_offset;
          indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + input_index] = output_coord_offset;
          // }
      }
  }
}
template <typename TIndiceUniq, typename TTable, typename TConvLocIter>
__global__ void calc_conv_indices_stage1_mask_direct_table(TTable table, TConvLocIter loc_iter, const int* indices_in, int32_t* indice_pairs_bwd, TIndiceUniq* indice_pairs_for_uniq, int* indice_num_per_loc, int num_indices_in, int RS, bool transposed)   {
  
  int filter_offset = blockIdx.y;
  loc_iter.set_filter_offset(filter_offset);
  // int indices_pair_size_mul_RS = num_indices_in * RS;
  int filter_offset_mul_indices_pair_size = filter_offset * num_indices_in;
  for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {
      tv::array<int, 4> npq_offset;
      bool valid;
      if (transposed){
          valid = loc_iter.query_nhw_out(indices_in + input_index * 4, npq_offset);
      }else{
          valid = loc_iter.query_npq(indices_in + input_index * 4, npq_offset);
      }
      if (valid){
          // int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
          TIndiceUniq output_coord_offset = loc_iter.layout_npq(npq_offset);
          // if (old_num < indices_pair_size){
          // indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;
          // indice_pairs_bwd[filter_offset_mul_indices_pair_size + input_index] = output_coord_offset;
          // indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + old_num] = output_coord_offset;
          table.insert_key_only(output_coord_offset);
          indice_pairs_for_uniq[filter_offset_mul_indices_pair_size + input_index] = output_coord_offset;
          // }
      }
  }
}
template <typename TTable, bool CheckValueValid>
__global__ void calc_conv_indices_stage2_mask(TTable table, int* indice_pairs_fwd, int* indice_pairs_bwd, const typename TTable::key_type* indice_pairs_uniq_before_sort, uint32_t* mask_fwd, uint32_t* mask_bwd, int num_indices_in, int num_indices_out, int mask_int_count)   {
  
  int filter_offset = blockIdx.y;
  int filter_pointer_offset = filter_offset / 32;
  uint32_t filter_mask_fwd = (1u << (filter_offset % 32));
  // TODO following rule for even kernel size is wrong. 
  // uint32_t filter_mask_bwd = (1u << (gridDim.y - 1 - filter_offset));
  auto indice_pairs_fwd_filter = indice_pairs_fwd + filter_offset * num_indices_out;
  auto indice_pairs_bwd_filter = indice_pairs_bwd + filter_offset * num_indices_in;
  auto indice_pairs_uniq_before_sort_filter = indice_pairs_uniq_before_sort + filter_offset * num_indices_in;
  for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {
     auto output_coord_offset = indice_pairs_uniq_before_sort_filter[input_index];
      if (output_coord_offset != std::numeric_limits<typename TTable::key_type>::max()){
          auto table_offset = table.lookup_offset(output_coord_offset);
          if (table_offset != -1){
              auto output_index = table.value_ptr()[table_offset];
              bool valid = CheckValueValid ? output_index >= 0 : true;
              if (valid){
                  atomicOr(mask_fwd + output_index * mask_int_count + filter_pointer_offset, filter_mask_fwd);
                  // atomicOr(mask_bwd + input_index, filter_mask_bwd);
                  indice_pairs_fwd_filter[output_index] = input_index;
                  if (indice_pairs_bwd != nullptr){
                      indice_pairs_bwd_filter[input_index] = output_index;
                  }
              }
          }
      }
  }
}
/**
 * @param indice_pairs_bwd 
 * @param mask_bwd 
 * @param num_indices_in 
 * @param kv 
 * @param mask_int_count 
 */
__global__ void calc_conv_indices_stage2_mask_output(int* indice_pairs_bwd, uint32_t* mask_bwd, int num_indices_in, int kv, int mask_int_count);
template <typename TTable, bool CheckValueValid>
__global__ void calc_conv_indices_stage2_inference_mask(TTable table, int* indice_pairs_fwd, int* indice_pairs_bwd, const typename TTable::key_type* indice_pairs_uniq_before_sort, uint32_t* mask_fwd, int num_indices_in, int num_indices_out, int mask_int_count)   {
  
  int filter_offset = blockIdx.y;
  int filter_pointer_offset = filter_offset / 32;
  uint32_t filter_mask_fwd = (1u << (filter_offset % 32));
  auto indice_pairs_fwd_filter = indice_pairs_fwd + filter_offset * num_indices_out;
  // auto indice_pairs_bwd_filter = indice_pairs_bwd + filter_offset * num_indices_in;
  auto indice_pairs_uniq_before_sort_filter = indice_pairs_uniq_before_sort + filter_offset * num_indices_in;
  for (int input_index : tv::KernelLoopX<int>(num_indices_in)) {
      auto output_coord_offset = indice_pairs_uniq_before_sort_filter[input_index];
      if (output_coord_offset != std::numeric_limits<typename TTable::key_type>::max()){
          auto table_offset = table.lookup_offset(output_coord_offset);
          if (table_offset != -1){
              auto output_index = table.value_ptr()[table_offset];
              bool valid = CheckValueValid ? output_index >= 0 : true;
              if (valid){
                  atomicOr(mask_fwd + output_index * mask_int_count + filter_pointer_offset, filter_mask_fwd);
                  indice_pairs_fwd_filter[output_index] = input_index;
              }
          }
      }
  }
}
template <typename TTable, typename TLayoutNPQ>
__global__ void build_subm_conv_hash_table(TTable table, const int* indices_in, TLayoutNPQ layout_npq, int num_indices)   {
  
  for (int i : tv::KernelLoopX<int>(num_indices)) {
      table.insert(layout_npq(indices_in + i * 4), i);
  }
}
template <typename T>
__global__ void clean_indices_uniq(T* indice_pairs_for_uniq, size_t size)   {
  
  for (size_t i : tv::KernelLoopX<size_t>(size)) {
      indice_pairs_for_uniq[i] = std::numeric_limits<T>::max();
  }
}
template <typename TTable, typename TConvLocIter>
__global__ void calc_subm_conv_indices(TConvLocIter loc_iter, TTable table, const int* indices_in, int32_t* indice_pairs, int* indice_num_per_loc, int num_indices_in, int indices_pair_size, int RS)   {
  
  int filter_offset = blockIdx.y;
  loc_iter.set_filter_offset(filter_offset);
  int indices_pair_size_mul_RS = indices_pair_size * RS;
  int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
  int filter_offset_mul_indices_pair_size_1 = (RS - 1 - filter_offset) * indices_pair_size;
  if (filter_offset == (RS / 2)){
      for (int i : tv::KernelLoopX<int>(num_indices_in)) {
          indice_pairs[filter_offset_mul_indices_pair_size + i] = i;
          indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + i] = i;
      }
  } else {
      for (int i : tv::KernelLoopX<int>(num_indices_in)) {
          tv::array<int, 4> npq_offset;
          if (loc_iter.query_npq_no_stride(indices_in + i * 4, npq_offset)){
              auto offset = loc_iter.layout_npq(npq_offset);
              // auto item = table.lookup(offset); // performance bound
              auto table_offset = table.lookup_offset(offset); // performance bound
              if (table_offset != -1){
                  auto v = table.value_ptr()[table_offset];
                  int old_num = tv::cuda::atomicAggInc(indice_num_per_loc + filter_offset);
                  indice_pairs[filter_offset_mul_indices_pair_size + old_num] = i;
                  indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + old_num] = v;
                  indice_pairs[filter_offset_mul_indices_pair_size_1 + old_num] = v;
                  indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size_1 + old_num] = i;
              }
          }
      }
  }
}
template <typename TTable, typename TConvLocIter>
__global__ void calc_subm_conv_indices_mask(TConvLocIter loc_iter, TTable table, const int* indices_in, int32_t* indice_pairs, uint32_t* mask, int num_indices, int indices_pair_size, int RS, bool is_train, int mask_int_count = 1)   {
  
  int filter_offset = blockIdx.y;
  uint32_t filter_mask_out = (1u << (filter_offset % 32));
  uint32_t filter_mask_out_offset = filter_offset / 32;
  uint32_t filter_mask_in = (1u << ((RS - 1 - filter_offset) % 32));
  uint32_t filter_mask_in_offset = (RS - 1 - filter_offset) / 32;
  // uint32_t filter_mask_center = (1u << (RS / 2));
  loc_iter.set_filter_offset(filter_offset);
  int indices_pair_size_mul_RS = indices_pair_size * RS;
  int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
  int filter_offset_mul_indices_pair_size_1 = (RS - 1 - filter_offset) * indices_pair_size;
  if (filter_offset == (RS / 2)){
      for (int i : tv::KernelLoopX<int>(num_indices)) {
          // atomicOr(mask + i, filter_mask_center);
          indice_pairs[filter_offset_mul_indices_pair_size + i] = i;
          if (is_train){
              indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + i] = i;
          }
      }
  } else {
      for (int output_index : tv::KernelLoopX<int>(num_indices)) {
          // find input offset from output offset
          tv::array<int, 4> nhw_offset;
          // table: input indice coord to output index (or output indice coord to input index)
          if (loc_iter.query_nhw(indices_in + output_index * 4, nhw_offset)){
              auto offset = loc_iter.layout_npq(nhw_offset);
              // auto item = table.lookup(offset);
              auto table_offset = table.lookup_offset(offset); // performance bound
              if (table_offset != -1){
                  auto input_index = table.value_ptr()[table_offset]; // we find a input indice idx.
                  atomicOr(mask + output_index * mask_int_count + filter_mask_out_offset, filter_mask_out);
                  atomicOr(mask + input_index * mask_int_count + filter_mask_in_offset, filter_mask_in);
                  // for this output, we set correct input idx.
                  indice_pairs[filter_offset_mul_indices_pair_size + output_index] = input_index;
                  if (is_train){
                      indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size + input_index] = output_index;
                  }
                  // the output in "input location" connect this output idx in another location.
                  indice_pairs[filter_offset_mul_indices_pair_size_1 + input_index] = output_index;
                  if (is_train){
                      indice_pairs[indices_pair_size_mul_RS + filter_offset_mul_indices_pair_size_1 + output_index] = input_index;
                  }
              }
          }
      }
  }
}
template <typename TTable, typename TConvLocIter>
__global__ void calc_subm_conv_indices_split_mask(TConvLocIter loc_iter, TTable table, const int* indices_in, int32_t* indice_pairs, uint32_t* mask1, uint32_t* mask2, int num_indices, int indices_pair_size, int RS, bool is_train)   {
  
  int filter_offset = blockIdx.y;
  uint32_t filter_mask_out = (1u << (filter_offset));
  uint32_t filter_mask_in = (1u << (RS - 1 - filter_offset));
  // uint32_t filter_mask_center = (1u << (RS / 2));
  loc_iter.set_filter_offset(filter_offset);
  auto indice_ptr_inv = indice_pairs + indices_pair_size * RS;
  int filter_offset_mul_indices_pair_size = filter_offset * indices_pair_size;
  int filter_offset_mul_indices_pair_size_1 = (RS - 1 - filter_offset) * indices_pair_size;
  if (filter_offset == (RS / 2)){
      for (int i : tv::KernelLoopX<int>(num_indices)) {
          indice_pairs[filter_offset_mul_indices_pair_size + i] = i;
          if (is_train){
              indice_ptr_inv[filter_offset_mul_indices_pair_size + i] = i;
          }
      }
  } else {
      for (int output_index : tv::KernelLoopX<int>(num_indices)) {
          // find input offset from output offset
          tv::array<int, 4> nhw_offset;
          // table: input indice coord to output index (or output indice coord to input index)
          if (loc_iter.query_nhw(indices_in + output_index * 4, nhw_offset)){
              auto offset = loc_iter.layout_npq(nhw_offset);
              auto table_offset = table.lookup_offset(offset); // performance bound
              if (table_offset != -1){
                  auto input_index = table.value_ptr()[table_offset]; // we find a input indice idx.
                  atomicOr(mask1 + output_index, filter_mask_out);
                  atomicOr(mask2 + input_index, filter_mask_in);
                  // for this output, we set correct input idx.
                  indice_pairs[filter_offset_mul_indices_pair_size + output_index] = input_index;
                  // the output in "input location" connect this output idx in another location.
                  indice_pairs[filter_offset_mul_indices_pair_size_1 + input_index] = output_index;
                  if (is_train){
                      indice_ptr_inv[filter_offset_mul_indices_pair_size + input_index] = output_index;
                      indice_ptr_inv[filter_offset_mul_indices_pair_size_1 + output_index] = input_index;
                  }
              }
          }
      }
  }
}
template <typename T>
__global__ void init_subm_multiple_mask_int_kernel(T* ptr, int set_bit, int length, int mask_int_count)   {
  
  int initial_offset = blockIdx.x * blockDim.x + threadIdx.x;
  int bit_offset = set_bit / 32;
  int bit_residue = set_bit % 32;
  for(int offset : tv::KernelLoopX<int>(length)){
      for (int i=0; i < mask_int_count; ++i)
          ptr[offset * mask_int_count + i] = (i == bit_offset) * (1 << bit_residue);
  }
}
struct SparseConvIndicesKernel {
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
  static void generate_conv_inds_stage1(tv::Tensor indices, tv::Tensor indice_pairs, tv::Tensor indice_pairs_uniq, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
  /**
   * @param indice_pairs_uniq 
   * @param uniq_size 
   * @param stream_int 
   */
  static int generate_conv_inds_stage1_5(tv::Tensor indice_pairs_uniq, int64_t uniq_size, std::uintptr_t stream_int = 0);
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
  static int generate_conv_inds_stage2(tv::Tensor indices, tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor indice_pairs, tv::Tensor indice_pairs_uniq, tv::Tensor indice_pairs_uniq_before_sort, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int num_out_act, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, bool transposed = false, std::uintptr_t stream_int = 0, bool use_bound_algo = false);
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
  static void generate_conv_inds_mask_stage1(tv::Tensor indices, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
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
  static void generate_conv_inds_mask_stage1_direct_table(tv::Tensor indices, tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
  /**
   * here indice_pairs_uniq may be bounded, some
   * points may be dropped.
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
  static int generate_conv_inds_stage2_mask(tv::Tensor indices, tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor indice_pairs_fwd, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor indice_pairs_uniq_before_sort, tv::Tensor out_inds, tv::Tensor mask_fwd, tv::Tensor mask_bwd, int num_out_act, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
  /**
   * here indice_pairs_uniq may be bounded, some
   * points may be dropped.
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
  static int generate_conv_inds_stage2_mask_direct_table(tv::Tensor indices, tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor indice_pairs_fwd, tv::Tensor indice_pairs_bwd, tv::Tensor indice_pairs_uniq, tv::Tensor indice_pairs_uniq_before_sort, tv::Tensor out_inds, tv::Tensor mask_fwd, tv::Tensor mask_bwd, int num_out_act, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, bool transposed = false, std::uintptr_t stream_int = 0);
  /**
   * unique by hash
   *         
   * @param hashdata_k 
   * @param hashdata_v 
   * @param uniq_cnt 
   * @param out_inds 
   * @param num_out_bound 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param stream_int 
   */
  static int unique_and_assign_output_direct_hash(tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor uniq_cnt, tv::Tensor out_inds, int num_out_bound, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, std::uintptr_t stream_int = 0);
  /**
   * unique by hash
   *         
   * @param hashdata_k 
   * @param hashdata_v 
   * @param uniq_cnt 
   * @param out_indices_offset 
   * @param num_out_bound 
   * @param stream_int 
   */
  static int unique_hash(tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor uniq_cnt, tv::Tensor out_indices_offset, int num_out_bound, std::uintptr_t stream_int = 0);
  /**
   * unique by hash
   *         
   * @param out_indices_offset 
   * @param out_inds 
   * @param batch_size 
   * @param output_dims 
   * @param input_dims 
   * @param ksize 
   * @param stride 
   * @param padding 
   * @param dilation 
   * @param stream_int 
   */
  static void assign_output_direct_hash(tv::Tensor out_indices_offset, tv::Tensor out_inds, int batch_size, tv::array<int, 3> output_dims, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> stride, tv::array<int, 3> padding, tv::array<int, 3> dilation, std::uintptr_t stream_int = 0);
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
   * @param is_train 
   * @param stream_int 
   */
  static int generate_subm_conv_inds(tv::Tensor indices, tv::Tensor hashdata_k, tv::Tensor hashdata_v, tv::Tensor indice_pairs, tv::Tensor out_inds, tv::Tensor indice_num_per_loc, int batch_size, tv::array<int, 3> input_dims, tv::array<int, 3> ksize, tv::array<int, 3> dilation, tv::Tensor indice_pair_mask = tv::Tensor(), bool is_train = true, std::uintptr_t stream_int = 0);
};
} // namespace ops3d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib