#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
std::tuple<int, int, int> GemmMainUnitTest::extract_mnk(std::vector<int64_t> a_shape, std::vector<int64_t> b_shape, bool trans_a, bool trans_b, bool trans_c, int shuffle_type, std::vector<int64_t> a_inds_shape, std::vector<int64_t> b_inds_shape, std::vector<int64_t> c_inds_shape)   {
  
  if (trans_c) {
      trans_a = !trans_a;
      trans_b = !trans_b;
      std::swap(trans_a, trans_b);
      std::swap(a_shape, b_shape);
  }
  int m, n, k, k2;
  if (shuffle_type == 1){
      TV_ASSERT_RT_ERR(!trans_a, "a of shuffle AB must be row major");
      TV_ASSERT_RT_ERR(!c_inds_shape.empty(), "c_inds must not empty");
      if (!a_inds_shape.empty()){
          m = a_inds_shape[0];
      }else{
          m = a_shape[0];
      }
      k = a_shape[int(!trans_a)];
      k2 = b_shape[(int(trans_b))];
      n = b_shape[(int(!trans_b) )];
  }
  else if (shuffle_type == 2){
      TV_ASSERT_RT_ERR(!a_inds_shape.empty() && !b_inds_shape.empty(), "a_inds and c_inds must not empty");
      TV_ASSERT_RT_ERR(trans_a && !trans_b, "shuffle AB must be nt, i.e. backward weight");
      m = a_shape[(int(trans_a))];
      k = a_inds_shape[0];
      k2 = b_inds_shape[0];
      n = b_shape[(int(!trans_b) )];
  }
  else{
      m = a_shape[int(trans_a)];
      k = a_shape[(int(!trans_a))];
      k2 = b_shape[(int(trans_b))];
      n = b_shape[(int(!trans_b) )];
  }
  return std::make_tuple(m, n, k);
}
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib