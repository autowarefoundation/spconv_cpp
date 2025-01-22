#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_s8s8s8s32f32tnt_m32n32k32m16n16k32A1T8816_200_C301LLL_SK_S8 {
namespace mma {
namespace mma_ns_wa {
namespace layout {
struct MyTensorOpLayout {
  __forceinline__ __host__ __device__ constexpr  MyTensorOpLayout()   {
    
  }
  __forceinline__ __host__ __device__ constexpr static MyTensorOpLayout from_shape(const tv::array<int, 2> & shape)   {
    return MyTensorOpLayout();
  }
  __forceinline__ __host__ __device__ constexpr int64_t operator()(int32_t s, int32_t ec)  const {
    
    int vc = ec / 16;
    int interleaved_s = s / 4;
    int idx_in_interleave_s = s % 4;
    // shape_before_interleave = 8 // 4
    // int sw_idx_s = interleaved_s / 2;
    int sw_idx_c = vc / 2;
    int idx_in_sw_c = vc % 2 + idx_in_interleave_s * 2;
    int idx_in_sw_s = interleaved_s % 2;
    int subsw_idx_s = idx_in_sw_s / 2;
    int subsw_idx_c = idx_in_sw_c / 2;
    int idx_in_subsw_s = idx_in_sw_s % 2;
    int idx_in_subsw_c = idx_in_sw_c % 2;
    // if subsw_idx_s == 0, permuted_subsw_idx_c = 0/1 = subsw_idx_c
    // else permuted_subsw_idx_c = 1/0
    int permuted_subsw_idx_c = subsw_idx_c;
    if (1 > 1){
        permuted_subsw_idx_c = subsw_idx_c ^ (subsw_idx_s % 2);
    }
    int premuted_idx_in_subsw_c = idx_in_subsw_c ^ (idx_in_subsw_s % 2);
    int final_c = sw_idx_c * 8 + permuted_subsw_idx_c * 2 + premuted_idx_in_subsw_c;
    // if ec % epa != 0
    int final_ec = final_c * 16 + ec % 16;
    int final_s = interleaved_s * 256;
    return final_ec + final_s;
  }
  template <int LdmCountStride, int LdmCountContig>
  __forceinline__ __host__ __device__ constexpr static int64_t get_ldm_initial_offset(int32_t lane_idx, int32_t permute_m_pointer_idx, bool transpose)   {
    
    int stride = -1;
    int contig_vec = -1;
    if (LdmCountContig == 1){
        stride = lane_idx >> 2;
        contig_vec = ((lane_idx & 3) << 1) ^ (stride & 1) ^ permute_m_pointer_idx;
    } else if (LdmCountContig == 2 && LdmCountStride == 2){
        if (transpose){
            stride = ((lane_idx & 0b111) >> 2) + (lane_idx >> 4 << 1);
            contig_vec = ((lane_idx & 3) << 1) + (stride & 1) ^ ((lane_idx & 0b1000) >> 3);
        }else{
            stride = (lane_idx & 0b1111) >> 2;
            contig_vec = ((lane_idx & 3) << 1) + (stride & 1) ^ ((lane_idx & 0b10000) >> 4);
        }
    }else{
        stride = (lane_idx & 0b111) >> 2;
        contig_vec = ((lane_idx & 3) << 1) + (stride & 1) ^ ((lane_idx & 0b1000) >> 3) + (lane_idx >> 4 << 3);
    }
    return stride * 256 + contig_vec * 16;
  }
  template <int LdmCountStride, int LdmCountContig>
  __forceinline__ __host__ __device__ constexpr int64_t get_ldm_initial_offset_ref_cpp(int32_t lane_idx, int32_t permute_m_pointer_idx, bool transpose)   {
    
    int stride = -1;
    int contig_vec = -1;
    if (LdmCountStride == 1){
        stride = lane_idx & 0b111;
        contig_vec = lane_idx >> 3;
    } else if (LdmCountContig == 2 && LdmCountStride == 2){
        if (transpose){
            stride = (lane_idx & 0b111) + ((lane_idx >> 4) << 3);
            contig_vec = (lane_idx & 0b1111) >> 3;
        }else{
            stride = lane_idx & 0b1111;
            contig_vec = lane_idx >> 4;
        }
    }else{
        stride = lane_idx;
        contig_vec = 0;
    }
    return (*this)(stride, contig_vec * 16 + LdmCountContig * permute_m_pointer_idx * 16);
  }
};
} // namespace layout
} // namespace mma_ns_wa
} // namespace mma
} // namespace Turing_s8s8s8s32f32tnt_m32n32k32m16n16k32A1T8816_200_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib