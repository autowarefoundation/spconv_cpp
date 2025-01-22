#include <spconvlib/spconv/csrc/sparse/convops/gemmops/GemmTunerSimple.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
namespace gemmops {
using static_key_t = std::tuple<bool, bool, bool, int, int, int, int>;
using algo_cache_key_t = std::tuple<int, int, int, int, int>;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using GemmTuneResult = spconvlib::spconv::csrc::sparse::convops::GemmTuneResult;
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using CompileInfo = spconvlib::cumm::common::CompileInfo;
using GemmMain = spconvlib::cumm::gemm::main::GemmMainUnitTest;
std::vector<tv::gemm::GemmAlgoDesp> GemmTunerSimple::get_all_available(tv::Tensor a, tv::Tensor b, tv::Tensor c, bool trans_a, bool trans_b, bool trans_c, std::tuple<int, int> arch, int shuffle_type, bool use_tf32)   {
  
  if (trans_c){
      trans_a = !trans_a;
      trans_b = !trans_b;
      std::swap(trans_a, trans_b);
      std::swap(a, b);
      trans_c = false;
  }
  // auto avail_algos = get_available_algo_str_from_arch(arch);
  std::vector<tv::gemm::GemmAlgoDesp> finally_algos;
  static_key_t static_key = std::make_tuple(trans_a, trans_b, trans_c, int(a.dtype()),
      int(b.dtype()), int(c.dtype()), shuffle_type);
  if (static_key_to_desps_.find(static_key) == static_key_to_desps_.end()){
      return finally_algos;
  }
  auto& desps = static_key_to_desps_.at(static_key);
  for (auto& desp : desps){
      if (arch < desp.min_arch){
          continue;
      }
      if (arch >= std::make_tuple(7, 5) && desp.algo == "Volta"){
          continue;
      }
      if (!use_tf32){
          if (desp.tensorop[0] > 0 && a.dtype() == tv::float32 && b.dtype() == tv::float32){
              // tf32 op
              continue;
          }
      }
      auto lda = a.stride(0);
      auto ldb = b.stride(0);
      auto ldc = c.stride(0);
      if (desp.supported_ldx(lda, ldb, ldc)){
          auto desp2 = desp;
          if (desp.is_nvrtc){
              if (!CompileInfo::algo_can_be_nvrtc_compiled(desp.min_arch)){
                  continue;
              }
          }
          if (!CompileInfo::arch_is_compiled_gemm(arch)){
              if (!CompileInfo::gemm_algo_can_use_ptx(desp.min_arch, arch)){
                  if (CompileInfo::algo_can_be_nvrtc_compiled(desp.min_arch)){
                      desp2.is_nvrtc = true;
                  }else{
                      continue;
                  }
              }
          }
          finally_algos.push_back(desp2);
      }
  }
  std::sort(finally_algos.begin(), finally_algos.end(), [](auto a, auto b){return a.min_arch > b.min_arch;});
  return finally_algos;
}
} // namespace gemmops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib