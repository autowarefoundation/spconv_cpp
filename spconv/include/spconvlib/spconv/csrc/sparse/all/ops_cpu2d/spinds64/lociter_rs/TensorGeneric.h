#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops_cpu2d {
namespace spinds64 {
namespace lociter_rs {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct TensorGeneric {
  tv::array<int32_t, 1> strides;
  TV_HOST_DEVICE_INLINE  TensorGeneric(tv::array<int32_t, 1> const& strides) : strides(strides)  {
    
  }
  TV_HOST_DEVICE_INLINE static TensorGeneric from_shape(const tv::array<int, 2> & shape)   {
    
    return TensorGeneric({
    shape[1]
    });
  }
  TV_HOST_DEVICE_INLINE int64_t operator()(const tv::array<int, 2> & indexes)  const {
    
    return indexes[1] + int64_t(strides[0] * indexes[0]);
  }
  TV_HOST_DEVICE_INLINE int64_t operator()(const int* indexes)  const {
    
    return indexes[1] + int64_t(strides[0] * indexes[0]);
  }
  TV_HOST_DEVICE_INLINE int64_t operator()(const tv::array<int64_t, 2> & indexes)  const {
    
    return indexes[1] + int64_t(strides[0] * indexes[0]);
  }
  TV_HOST_DEVICE_INLINE int64_t operator()(const int64_t* indexes)  const {
    
    return indexes[1] + int64_t(strides[0] * indexes[0]);
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 2> inverse(int64_t index)  const {
    
    tv::array<int, 2> out;
    int64_t residual = index;
    out[0] = int(residual / strides[0]);
    out[1] = int(residual % strides[0]);
    return out;
  }
  TV_HOST_DEVICE_INLINE void inverse(int64_t index, tv::array<int, 2>& out)  const {
    
    int64_t residual = index;
    out[0] = int(residual / strides[0]);
    out[1] = int(residual % strides[0]);
  }
  TV_HOST_DEVICE_INLINE void inverse(int64_t index, int & idx_0, int & idx_1)  const {
    
    int64_t residual = index;
    idx_0 = int(residual / strides[0]);
    idx_1 = int(residual % strides[0]);
  }
  TV_HOST_DEVICE_INLINE void inverse(int64_t index, int* out)  const {
    
    int64_t residual = index;
    out[0] = int(residual / strides[0]);
    out[1] = int(residual % strides[0]);
  }
};
} // namespace lociter_rs
} // namespace spinds64
} // namespace ops_cpu2d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib