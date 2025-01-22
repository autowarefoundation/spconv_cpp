#pragma once
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/cumm/common/GemmBasic.h>
#include <spconvlib/cumm/common/GemmBasicHost.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
struct GemmMainUnitTest {
  /**
   * @param params 
   */
  static void matmul_split_Simt_f32f32f32_1(tv::gemm::GemmParams params);
  /**
   * @param params 
   */
  static void matmul_split_Simt_f16f16f16_1(tv::gemm::GemmParams params);
  /**
   * @param params 
   */
  static void matmul_split_Volta_f16f16f16_1(tv::gemm::GemmParams params);
  /**
   * @param params 
   */
  static void matmul_split_Turing_f16f16f16_1(tv::gemm::GemmParams params);
  
  static std::vector<tv::gemm::GemmAlgoDesp> get_all_algo_desp();
  /**
   * @param a_shape 
   * @param b_shape 
   * @param trans_a 
   * @param trans_b 
   * @param trans_c 
   * @param shuffle_type 
   * @param a_inds_shape 
   * @param b_inds_shape 
   * @param c_inds_shape 
   */
  static std::tuple<int, int, int> extract_mnk(std::vector<int64_t> a_shape, std::vector<int64_t> b_shape, bool trans_a, bool trans_b, bool trans_c, int shuffle_type = 0, std::vector<int64_t> a_inds_shape = std::vector<int64_t>{}, std::vector<int64_t> b_inds_shape = std::vector<int64_t>{}, std::vector<int64_t> c_inds_shape = std::vector<int64_t>{});
  /**
   * @param val 
   */
  static int align_to_power2(int val);
  
  static void device_synchronize();
  /**
   * @param stream 
   */
  static void stream_synchronize(std::uintptr_t stream);
  /**
   * @param m 
   * @param n 
   * @param k 
   * @param tile_ms 
   * @param tile_ns 
   * @param tile_ks 
   * @param tile_shape_to_algos 
   * @param large_k_first 
   */
  static std::vector<int> simple_select_tile_shape(int m, int n, int k, std::vector<int> tile_ms, std::vector<int> tile_ns, std::vector<int> tile_ks, std::unordered_map<int64_t, std::vector<int>> tile_shape_to_algos, bool large_k_first);
  /**
   * @param params 
   */
  static void matmul2(tv::gemm::GemmParams params);
};
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib