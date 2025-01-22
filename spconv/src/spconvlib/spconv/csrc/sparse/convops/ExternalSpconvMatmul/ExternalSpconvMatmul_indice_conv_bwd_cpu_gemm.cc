#include <spconvlib/spconv/csrc/sparse/convops/ExternalSpconvMatmul.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace convops {
using TensorView = spconvlib::cumm::common::TensorView;
void ExternalSpconvMatmul::indice_conv_bwd_cpu_gemm(std::string inp_buffer_n, std::string out_buffer_n, std::string filters_n, std::string dfilters_n, bool all_weight_is_krsc, bool is_kc_not_ck, int nhot, int index)   {
  
  TV_THROW_RT_ERR("not implemented, override this and use preferred cpu blas!!!");
}
} // namespace convops
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib