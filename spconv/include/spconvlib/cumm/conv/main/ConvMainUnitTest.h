#pragma once
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/cumm/common/GemmBasic.h>
#include <spconvlib/cumm/common/GemmBasicHost.h>
#include <spconvlib/cumm/conv/kernel/ConvNVRTCParams.h>
#include <spconvlib/cumm/common/CummNVRTCLib.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using ConvNVRTCParams = spconvlib::cumm::conv::kernel::ConvNVRTCParams;
using CummNVRTCLib = spconvlib::cumm::common::CummNVRTCLib;
struct ConvMainUnitTest {
  /**
   * @param params 
   */
  static void matmul_split_Simt_f32f32f32_0(tv::gemm::ConvParams params);
  /**
   * @param params 
   */
  static void matmul_split_Simt_f16f16f16_0(tv::gemm::ConvParams params);
  /**
   * @param params 
   */
  static void matmul_split_Volta_f16f16f16_0(tv::gemm::ConvParams params);
  /**
   * @param params 
   */
  static void matmul_split_Turing_s8s8s8_0(tv::gemm::ConvParams params);
  /**
   * @param params 
   */
  static void matmul_split_Turing_s8s8f32_0(tv::gemm::ConvParams params);
  /**
   * @param params 
   */
  static void matmul_split_Turing_s8s8f16_0(tv::gemm::ConvParams params);
  /**
   * @param params 
   */
  static void matmul_split_Turing_f16f16f16_0(tv::gemm::ConvParams params);
  /**
   * @param params 
   */
  static void matmul_split_Ampere_f32f32f32_0(tv::gemm::ConvParams params);
  /**
   * @param params 
   */
  static void matmul_split_Ampere_f16f16f16_0(tv::gemm::ConvParams params);
  /**
   * @param params 
   */
  static void matmul_split_Ampere_s8s8s8_0(tv::gemm::ConvParams params);
  /**
   * @param params 
   */
  static void matmul_split_Ampere_s8s8f32_0(tv::gemm::ConvParams params);
  /**
   * @param params 
   */
  static void matmul_split_Ampere_s8s8f16_0(tv::gemm::ConvParams params);
  /**
   * @param op_type 
   * @param N 
   * @param C 
   * @param K 
   * @param kernel_volume 
   * @param in_prod 
   * @param out_prod 
   * @param mask_sparse 
   */
  static std::array<int, 3> extract_mnk(int op_type, int N, int C, int K, int kernel_volume, int in_prod, int out_prod, bool mask_sparse);
  /**
   * @param params 
   */
  static void implicit_gemm2(tv::gemm::ConvParams params);
  
  static std::vector<tv::gemm::ConvAlgoDesp> get_all_conv_algo_desp();
};
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib