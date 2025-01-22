#include <spconvlib/spconv/csrc/utils/boxops/BoxOps.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace utils {
namespace boxops {
using TensorView = spconvlib::cumm::common::TensorView;
std::vector<int> BoxOps::non_max_suppression_cpu(tv::Tensor boxes, tv::Tensor order, float thresh, float eps)   {
  
  auto ndets = boxes.dim(0);
  std::vector<int> keep(ndets);
  tv::dispatch<float, double>(boxes.dtype(), [&](auto I1){
      using DType = TV_DECLTYPE(I1);
      auto boxes_r = boxes.tview<const DType, 2>();
      tv::dispatch<int, int64_t, uint32_t, uint64_t>(order.dtype(), [&](auto I2){
          using T2 = TV_DECLTYPE(I2);
          auto order_r = order.tview<const T2, 1>();
          std::vector<DType> areas;
          for (int i = 0; i < ndets; ++i){
              areas[i] = (boxes_r(i, 2) - boxes_r(i, 0) + eps) * 
                         (boxes_r(i, 3) - boxes_r(i, 1) + eps);
          }
          std::vector<int> suppressed(ndets, 0);
          int i, j;
          DType xx1, xx2, w, h, inter, ovr;
          for (int _i = 0; _i < ndets; ++_i) {
              i = order_r(_i);
              if (suppressed[i] == 1)
                  continue;
              keep.push_back(i);
              for (int _j = _i + 1; _j < ndets; ++_j) {
                  j = order_r(_j);
                  if (suppressed[j] == 1)
                      continue;
                  xx2 = std::min(boxes_r(i, 2), boxes_r(j, 2));
                  xx1 = std::max(boxes_r(i, 0), boxes_r(j, 0));
                  w = xx2 - xx1 + eps;
                  if (w > 0) {
                      xx2 = std::min(boxes_r(i, 3), boxes_r(j, 3));
                      xx1 = std::max(boxes_r(i, 1), boxes_r(j, 1));
                      h = xx2 - xx1 + eps;
                      if (h > 0) {
                      inter = w * h;
                      ovr = inter / (areas[i] + areas[j] - inter);
                      if (ovr >= thresh)
                          suppressed[j] = 1;
                      }
                  }
              }
          }
      });
  });
  return keep;
}
} // namespace boxops
} // namespace utils
} // namespace csrc
} // namespace spconv
} // namespace spconvlib