#pragma once
#include <spconvlib/spconv/csrc/sparse/alloc/ExternalAllocator.h>
#include <spconvlib/spconv/csrc/sparse/convops/GemmTuneResult.h>
#include <spconvlib/spconv/csrc/sparse/convops/ConvTuneResult.h>
#include <spconvlib/spconv/csrc/sparse/convops/ExternalSpconvMatmul.h>
#include <spconvlib/spconv/csrc/sparse/inference/InferenceOps.h>
#include <spconvlib/spconv/csrc/sparse/convops/gemmops/GemmTunerSimple.h>
#include <spconvlib/spconv/csrc/sparse/convops/convops/ConvTunerSimple.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
namespace spops {
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using GemmTuneResult = spconvlib::spconv::csrc::sparse::convops::GemmTuneResult;
using ConvTuneResult = spconvlib::spconv::csrc::sparse::convops::ConvTuneResult;
using ExternalSpconvMatmul = spconvlib::spconv::csrc::sparse::convops::ExternalSpconvMatmul;
using InferenceOps = spconvlib::spconv::csrc::sparse::inference::InferenceOps;
using GemmTuner = spconvlib::spconv::csrc::sparse::convops::gemmops::GemmTunerSimple;
using ConvTuner = spconvlib::spconv::csrc::sparse::convops::convops::ConvTunerSimple;
struct ConvGemmOps {
  /**
   * @param index 
   */
  static std::tuple<int, int> get_compute_capability(int index = -1);
  /**
   * 1. this function need to take a out features
   * that from subm first mm.
   * 2. this function don't support CPU.
   * @param allocator 
   * @param ext_mm 
   * @param gemm_tuner 
   * @param all_w_is_krsc 
   * @param filter_hwio 
   * @param features 
   * @param filters 
   * @param indice_pairs 
   * @param indice_pair_num 
   * @param arch 
   * @param num_activate_out 
   * @param inverse 
   * @param subm 
   * @param algo 
   * @param stream_int 
   * @param bias 
   * @param act_alpha 
   * @param act_beta 
   * @param act_type 
   * @param use_tf32 
   */
  static void indice_conv(ExternalAllocator& allocator, ExternalSpconvMatmul& ext_mm, GemmTuner& gemm_tuner, bool all_w_is_krsc, bool filter_hwio, tv::Tensor features, tv::Tensor filters, tv::Tensor indice_pairs, tv::Tensor indice_pair_num, std::tuple<int, int> arch, int num_activate_out, bool inverse = false, bool subm = false, int algo = 0, std::uintptr_t stream_int = 0, tv::Tensor bias = tv::Tensor(), float act_alpha = 0.0, float act_beta = 0.0, tv::gemm::Activation act_type = tv::gemm::Activation::kNone, bool use_tf32 = true);
  /**
   * @param allocator 
   * @param ext_mm 
   * @param gemm_tuner 
   * @param all_w_is_krsc 
   * @param filter_hwio 
   * @param features 
   * @param filters 
   * @param out_bp 
   * @param indice_pairs 
   * @param indice_pair_num 
   * @param arch 
   * @param inverse 
   * @param subm 
   * @param algo 
   * @param stream_int 
   * @param use_tf32 
   */
  static void indice_conv_backward(ExternalAllocator& allocator, ExternalSpconvMatmul& ext_mm, GemmTuner& gemm_tuner, bool all_w_is_krsc, bool filter_hwio, tv::Tensor features, tv::Tensor filters, tv::Tensor out_bp, tv::Tensor indice_pairs, tv::Tensor indice_pair_num, std::tuple<int, int> arch, bool inverse = false, bool subm = false, int algo = 0, std::uintptr_t stream_int = 0, bool use_tf32 = true);
  /**
   * @param allocator 
   * @param conv_tuner 
   * @param features 
   * @param filters 
   * @param pair_fwd 
   * @param pair_mask_fwd_splits 
   * @param mask_argsort_fwd_splits 
   * @param num_activate_out 
   * @param masks 
   * @param arch 
   * @param is_train 
   * @param is_subm 
   * @param stream_int 
   * @param timer 
   * @param auto_fp32_accum 
   * @param fp32_accum 
   * @param bias 
   * @param act_alpha 
   * @param act_beta 
   * @param act_type 
   * @param use_tf32 
   * @param output_scale 
   * @param scale 
   * @param output_add 
   * @param output_add_scale 
   * @param output_dtype 
   */
  static std::tuple<int, ConvTuneResult> implicit_gemm(ExternalAllocator& allocator, ConvTuner& conv_tuner, tv::Tensor features, tv::Tensor filters, tv::Tensor pair_fwd, std::vector<tv::Tensor> pair_mask_fwd_splits, std::vector<tv::Tensor> mask_argsort_fwd_splits, int num_activate_out, tv::Tensor masks, std::tuple<int, int> arch, bool is_train = false, bool is_subm = false, std::uintptr_t stream_int = 0, tv::CUDAKernelTimer timer = tv::CUDAKernelTimer(false), bool auto_fp32_accum = true, bool fp32_accum = false, tv::Tensor bias = tv::Tensor(), float act_alpha = 0.0, float act_beta = 0.0, tv::gemm::Activation act_type = tv::gemm::Activation::kNone, bool use_tf32 = true, float output_scale = 1.0, tv::Tensor scale = tv::Tensor(), tv::Tensor output_add = tv::Tensor(), float output_add_scale = 1.0, int output_dtype = -1);
  /**
   * @param allocator 
   * @param conv_tuner 
   * @param features 
   * @param filters 
   * @param out_bp 
   * @param pair_fwd 
   * @param pair_bwd 
   * @param pair_mask_fwd_splits 
   * @param pair_mask_bwd_splits 
   * @param mask_argsort_fwd_splits 
   * @param mask_argsort_bwd_splits 
   * @param mask_output_fwd 
   * @param masks 
   * @param arch 
   * @param mask_width 
   * @param is_subm 
   * @param stream_int 
   * @param timer 
   * @param auto_fp32_accum 
   * @param fp32_accum 
   * @param use_tf32 
   */
  static void implicit_gemm_backward(ExternalAllocator& allocator, ConvTuner& conv_tuner, tv::Tensor features, tv::Tensor filters, tv::Tensor out_bp, tv::Tensor pair_fwd, tv::Tensor pair_bwd, std::vector<tv::Tensor> pair_mask_fwd_splits, std::vector<tv::Tensor> pair_mask_bwd_splits, std::vector<tv::Tensor> mask_argsort_fwd_splits, std::vector<tv::Tensor> mask_argsort_bwd_splits, tv::Tensor mask_output_fwd, tv::Tensor masks, std::tuple<int, int> arch, int mask_width, bool is_subm, std::uintptr_t stream_int = 0, tv::CUDAKernelTimer timer = tv::CUDAKernelTimer(false), bool auto_fp32_accum = true, bool fp32_accum = false, bool use_tf32 = true);
};
} // namespace spops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib