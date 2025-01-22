#include <spconvlib/spconv/csrc/utils/launch/LaunchUtils.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace utils {
namespace launch {
using TensorView = spconvlib::cumm::common::TensorView;
std::tuple<int, int, int, int> LaunchUtils::get_blocks_threads_of_2d_tensor(int64_t nhot, int64_t num_features)   {
  
  constexpr int MaxThreads = 512;
  int num_blocks_X = 0;
  int num_blocks_Y = 0;
  int threads_X = 0;
  int threads_Y = 0;
  dim3 threads;
  bool found = tv::dispatch_int_noexcept<512, 256, 128, 64, 32, 16>(int(num_features), [](int my, int expect){return my >= expect;}, [&](auto V){
      // if num_features > value in list above, run this function.
      // if a value is found, other value won't be executed.
      int NumFeatures = TV_DECLTYPE(V)::value;
      int Num0 = MaxThreads / NumFeatures;
      num_blocks_X = tv::div_up(num_features, int64_t(NumFeatures));
      num_blocks_Y = tv::div_up(nhot, int64_t(Num0));
      threads_X = NumFeatures;
      threads_Y = Num0;
  });
  if (!found){
      int NumFeatures = 16;
      int Num0 = MaxThreads / NumFeatures;
      num_blocks_X = tv::div_up(num_features, int64_t(NumFeatures));
      num_blocks_Y = tv::div_up(nhot, int64_t(Num0));
      threads_X = NumFeatures;
      threads_Y = Num0;
  }
  return std::make_tuple(num_blocks_X, num_blocks_Y, threads_X, threads_Y);
}
} // namespace launch
} // namespace utils
} // namespace csrc
} // namespace spconv
} // namespace spconvlib