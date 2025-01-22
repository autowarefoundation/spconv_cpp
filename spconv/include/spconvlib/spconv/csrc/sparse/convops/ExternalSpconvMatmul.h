#pragma once
#include <spconvlib/cumm/common/TensorView.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
using TensorView = spconvlib::cumm::common::TensorView;
struct ExternalSpconvMatmul {
  /**
   * @param features_n 
   * @param filters_n 
   * @param all_weight_is_krsc 
   * @param is_kc_not_ck 
   * @param kv_center 
   * @param out_channel 
   * @param stream_int 
   */
  virtual tv::Tensor indice_conv_init_gemm(std::string features_n, std::string filters_n, bool all_weight_is_krsc, bool is_kc_not_ck, int kv_center, int out_channel, std::uintptr_t stream_int = 0);
  /**
   * @param inp_buffer_n 
   * @param out_buffer_n 
   * @param filters_n 
   * @param all_weight_is_krsc 
   * @param is_kc_not_ck 
   * @param nhot 
   * @param index 
   */
  virtual void indice_conv_cpu_gemm(std::string inp_buffer_n, std::string out_buffer_n, std::string filters_n, bool all_weight_is_krsc, bool is_kc_not_ck, int nhot, int index);
  /**
   * @param features_n 
   * @param filters_n 
   * @param out_bp_n 
   * @param dfilters_n 
   * @param all_weight_is_krsc 
   * @param is_kc_not_ck 
   * @param kv_center 
   * @param stream_int 
   */
  virtual tv::Tensor indice_conv_bwd_init_gemm(std::string features_n, std::string filters_n, std::string out_bp_n, std::string dfilters_n, bool all_weight_is_krsc, bool is_kc_not_ck, int kv_center, std::uintptr_t stream_int = 0);
  /**
   * @param inp_buffer_n 
   * @param out_buffer_n 
   * @param filters_n 
   * @param dfilters_n 
   * @param all_weight_is_krsc 
   * @param is_kc_not_ck 
   * @param nhot 
   * @param index 
   */
  virtual void indice_conv_bwd_cpu_gemm(std::string inp_buffer_n, std::string out_buffer_n, std::string filters_n, std::string dfilters_n, bool all_weight_is_krsc, bool is_kc_not_ck, int nhot, int index);
};
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib