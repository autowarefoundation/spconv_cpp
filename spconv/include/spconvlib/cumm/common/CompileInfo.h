#pragma once
#include <vector>
#include <tuple>
#include <string>
namespace spconvlib {
namespace cumm {
namespace common {
struct CompileInfo {
  
  static std::tuple<int, int> get_compiled_cuda_version();
  
  static std::vector<std::tuple<int, int>> get_compiled_cuda_arch();
  
  static std::vector<std::tuple<int, int>> get_compiled_gemm_cuda_arch();
  /**
   * @param arch 
   */
  static bool arch_is_compiled(std::tuple<int, int> arch);
  /**
   * @param arch 
   */
  static bool arch_is_compiled_gemm(std::tuple<int, int> arch);
  /**
   * @param arch 
   */
  static bool arch_is_compatible(std::tuple<int, int> arch);
  /**
   * @param arch 
   */
  static bool arch_is_compatible_gemm(std::tuple<int, int> arch);
  /**
   * @param min_arch 
   * @param arch 
   */
  static bool algo_can_use_ptx(std::tuple<int, int> min_arch, std::tuple<int, int> arch);
  /**
   * @param min_arch 
   * @param arch 
   */
  static bool gemm_algo_can_use_ptx(std::tuple<int, int> min_arch, std::tuple<int, int> arch);
  /**
   * @param min_arch 
   */
  static bool algo_can_be_nvrtc_compiled(std::tuple<int, int> min_arch);
};
} // namespace common
} // namespace cumm
} // namespace spconvlib