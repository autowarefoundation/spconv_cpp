#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops_cpu3d {
namespace spinds {
namespace lociter {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct TensorGeneric {
  tv::array<int32_t, 3> strides;
  TV_HOST_DEVICE_INLINE  TensorGeneric(tv::array<int32_t, 3> const& strides) : strides(strides)  {
    
  }
  TV_HOST_DEVICE_INLINE static TensorGeneric from_shape(const tv::array<int, 4> & shape)   {
    
    return TensorGeneric({
    shape[1] * shape[2] * shape[3],
    shape[2] * shape[3],
    shape[3]
    });
  }
  TV_HOST_DEVICE_INLINE int64_t operator()(const tv::array<int, 4> & indexes)  const {
    
    return indexes[3] + int64_t(strides[0] * indexes[0]) + int64_t(strides[1] * indexes[1]) + int64_t(strides[2] * indexes[2]);
  }
  TV_HOST_DEVICE_INLINE int64_t operator()(const int* indexes)  const {
    
    return indexes[3] + int64_t(strides[0] * indexes[0]) + int64_t(strides[1] * indexes[1]) + int64_t(strides[2] * indexes[2]);
  }
  TV_HOST_DEVICE_INLINE int64_t operator()(const tv::array<int64_t, 4> & indexes)  const {
    
    return indexes[3] + int64_t(strides[0] * indexes[0]) + int64_t(strides[1] * indexes[1]) + int64_t(strides[2] * indexes[2]);
  }
  TV_HOST_DEVICE_INLINE int64_t operator()(const int64_t* indexes)  const {
    
    return indexes[3] + int64_t(strides[0] * indexes[0]) + int64_t(strides[1] * indexes[1]) + int64_t(strides[2] * indexes[2]);
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 4> inverse(int64_t index)  const {
    
    tv::array<int, 4> out;
    int64_t residual = index;
    out[0] = int(residual / strides[0]);
    residual = residual % strides[0];
    out[1] = int(residual / strides[1]);
    residual = residual % strides[1];
    out[2] = int(residual / strides[2]);
    out[3] = int(residual % strides[2]);
    return out;
  }
  TV_HOST_DEVICE_INLINE void inverse(int64_t index, tv::array<int, 4>& out)  const {
    
    int64_t residual = index;
    out[0] = int(residual / strides[0]);
    residual = residual % strides[0];
    out[1] = int(residual / strides[1]);
    residual = residual % strides[1];
    out[2] = int(residual / strides[2]);
    out[3] = int(residual % strides[2]);
  }
  TV_HOST_DEVICE_INLINE void inverse(int64_t index, int & idx_0, int & idx_1, int & idx_2, int & idx_3)  const {
    
    int64_t residual = index;
    idx_0 = int(residual / strides[0]);
    residual = residual % strides[0];
    idx_1 = int(residual / strides[1]);
    residual = residual % strides[1];
    idx_2 = int(residual / strides[2]);
    idx_3 = int(residual % strides[2]);
  }
  TV_HOST_DEVICE_INLINE void inverse(int64_t index, int* out)  const {
    
    int64_t residual = index;
    out[0] = int(residual / strides[0]);
    residual = residual % strides[0];
    out[1] = int(residual / strides[1]);
    residual = residual % strides[1];
    out[2] = int(residual / strides[2]);
    out[3] = int(residual % strides[2]);
  }
};
} // namespace lociter
} // namespace spinds
} // namespace ops_cpu3d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib