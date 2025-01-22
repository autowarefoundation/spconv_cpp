#include <spconvlib/spconv/csrc/sparse/convops/convops/ConvTunerSimple.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
namespace convops {
using static_key_t = std::tuple<int, int, int, int, int, int, int, int, int, int>;
using algo_cache_key_t = std::tuple<int, int, int, int, int, int, int, int, bool>;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using ConvTuneResult = spconvlib::spconv::csrc::sparse::convops::ConvTuneResult;
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using CompileInfo = spconvlib::cumm::common::CompileInfo;
using ConvMain = spconvlib::cumm::conv::main::ConvMainUnitTest;
std::tuple<ConvTuneResult, bool> ConvTunerSimple::get_tuned_algo(int op_type, int i_dtype, int w_dtype, int o_dtype, int k, int c, std::tuple<int, int> arch, int mask_width, bool need_dynamic_mask)   {
  
  tv::gemm::ConvOpType op_type_cpp = static_cast<tv::gemm::ConvOpType>(op_type);
  if (op_type_cpp != tv::gemm::ConvOpType::kBackwardWeight){
      mask_width = -1;
  }
  algo_cache_key_t key;
  key = std::make_tuple(i_dtype, w_dtype, o_dtype, k, c, 
      std::get<0>(arch), std::get<1>(arch), mask_width, need_dynamic_mask);
  ConvTuneResult res;
  bool exists = false;
  {
      std::lock_guard<std::mutex> guard(mutex_);
      if (op_type_cpp == tv::gemm::ConvOpType::kForward){
          if (kc_forward_cache_.find(key) != kc_forward_cache_.end()){
              res = kc_forward_cache_.at(key);
              exists = true;
          }
      }
      else if (op_type_cpp == tv::gemm::ConvOpType::kBackwardInput){
          if (kc_dgrad_cache_.find(key) != kc_dgrad_cache_.end()){
              res = kc_dgrad_cache_.at(key);
              exists = true;
          }
      }
      else if (op_type_cpp == tv::gemm::ConvOpType::kBackwardWeight){
          if (kc_wgrad_cache_.find(key) != kc_wgrad_cache_.end()){
              res = kc_wgrad_cache_.at(key);
              exists = true;
          }
      }
      else{
          TV_THROW_RT_ERR("not implemented");
      }
  }
  return std::make_tuple(res, exists);
}
} // namespace convops
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib