#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTC.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops1d {
namespace spinds {
namespace lociter_rs {
using TensorViewNVRTC = spconvlib::cumm::common::TensorViewNVRTC;
struct TensorGeneric {
  TV_HOST_DEVICE_INLINE  TensorGeneric()   {
    
  }
  TV_HOST_DEVICE_INLINE static TensorGeneric from_shape(const tv::array<int, 1> & shape)   {
    
    return TensorGeneric();
  }
  TV_HOST_DEVICE_INLINE int64_t operator()(const tv::array<int, 1> & indexes)  const {
    
    return indexes[0];
  }
  TV_HOST_DEVICE_INLINE int64_t operator()(const int* indexes)  const {
    
    return indexes[0];
  }
  TV_HOST_DEVICE_INLINE int64_t operator()(const tv::array<int64_t, 1> & indexes)  const {
    
    return indexes[0];
  }
  TV_HOST_DEVICE_INLINE int64_t operator()(const int64_t* indexes)  const {
    
    return indexes[0];
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 1> inverse(int64_t index)  const {
    
    tv::array<int, 1> out;
    out[0] = index;
    return out;
  }
  TV_HOST_DEVICE_INLINE void inverse(int64_t index, tv::array<int, 1>& out)  const {
    
    out[0] = index;
  }
  TV_HOST_DEVICE_INLINE void inverse(int64_t index, int & idx_0)  const {
    
    idx_0 = index;
  }
  TV_HOST_DEVICE_INLINE void inverse(int64_t index, int* out)  const {
    
    out[0] = index;
  }
};
} // namespace lociter_rs
} // namespace spinds
} // namespace ops1d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib