#include <spconvlib/spconv/csrc/sparse/convops/SimpleExternalSpconvMatmul.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
using TensorView = spconvlib::cumm::common::TensorView;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using ExternalSpconvMatmul = spconvlib::spconv::csrc::sparse::convops::ExternalSpconvMatmul;
tv::Tensor SimpleExternalSpconvMatmul::indice_conv_init_gemm(std::string features_n, std::string filters_n, bool all_weight_is_krsc, bool is_kc_not_ck, int kv_center, int out_channel, std::uintptr_t stream_int)   {
  
  auto features = alloc_.get_tensor_by_name(features_n);
  auto filters = alloc_.get_tensor_by_name(filters_n);
  TV_ASSERT_RT_ERR(!features.is_cpu(), "only supprt cuda");
  auto out_features = alloc_.empty("OutFeatures", 
      {features.dim(0), out_channel}, features.dtype(), features.device());
  if (!all_weight_is_krsc){
      filters = filters.view(-1, filters.dim(-2), filters.dim(-1));
      if (!is_kc_not_ck){
          matmul(handle_, reinterpret_cast<cudaStream_t>(stream_int), 
              features, filters[kv_center], out_features, false, false);
      }else{
          matmul(handle_, reinterpret_cast<cudaStream_t>(stream_int), 
              features, filters[kv_center], out_features, false, true);
      }
  }else{
      filters = filters.view(out_channel, -1, filters.dim(-1));
      matmul(handle_, reinterpret_cast<cudaStream_t>(stream_int), 
          features, filters.select(1, kv_center), out_features, false, true);
  }
  return out_features;
}
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib