#pragma once
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu1d/spinds/ConvProblem.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu1d/spinds/lociter/TensorGeneric.h>
#include <spconvlib/spconv/csrc/sparse/all/ops_cpu1d/spinds/lociter_rs/TensorGeneric.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
namespace ops_cpu1d {
namespace spinds {
using TensorView = spconvlib::cumm::common::TensorView;
using ConvProblem = spconvlib::spconv::csrc::sparse::all::ops_cpu1d::spinds::ConvProblem;
using LayoutNPQ = spconvlib::spconv::csrc::sparse::all::ops_cpu1d::spinds::lociter::TensorGeneric;
using LayoutRS = spconvlib::spconv::csrc::sparse::all::ops_cpu1d::spinds::lociter_rs::TensorGeneric;
struct ConvOutLocIter {
  ConvProblem problem_;
  tv::array<int, 1> count_;
  LayoutNPQ layout_npq;
  LayoutRS layout_rs;
  TV_HOST_DEVICE_INLINE  ConvOutLocIter(ConvProblem const& problem) : problem_(problem), count_({0}), layout_npq(LayoutNPQ::from_shape({problem.N, problem.output_dims[0]})), layout_rs(LayoutRS::from_shape({problem.ksize[0]}))  {
    
  }
  TV_HOST_DEVICE_INLINE ConvOutLocIter& operator++()   {
    
    if (++count_[0] < problem_.ksize[0]){
        return *this;
    }
    count_[0] = 0;
    return *this;
  }
  TV_HOST_DEVICE_INLINE void set_filter_offset(int filter_offset)   {
    
    layout_rs.inverse(filter_offset, count_);
  }
  template <bool NoStride>
  TV_HOST_DEVICE_INLINE tv::array<int, 2> nhw_to_npq(const int* nhw_offset)  const {
    
    int r_0 = count_[0];
    int h_0 = (nhw_offset[1] + problem_.padding[0] - 
        r_0 * problem_.dilation[0]) / (NoStride ? 1 : problem_.stride[0]);
    return {nhw_offset[0], h_0};
  }
  TV_HOST_DEVICE_INLINE tv::array<int, 2> npq_to_nhw(const int* npq_offset)  const {
    
    int r_0 = count_[0];
    int h_0 = npq_offset[1] * problem_.stride[0] - problem_.padding[0] + r_0 * problem_.dilation[0];
    return {npq_offset[0], h_0};
  }
  TV_HOST_DEVICE_INLINE bool query_npq(const int* nhw_offset, tv::array<int, 2>& npq_offset)  const {
    
    auto npq_no_stride = nhw_to_npq<true>(nhw_offset);
    npq_offset[0] = npq_no_stride[0];
    npq_offset[1] = npq_no_stride[1] / problem_.stride[0];
    return (npq_no_stride[0] < problem_.N) && 
        (npq_no_stride[0] >= 0) && 
        npq_offset[1] >= 0 && npq_offset[1] < problem_.output_dims[0] &&
        !(npq_no_stride[1] % problem_.stride[0]);
  }
  TV_HOST_DEVICE_INLINE bool query_npq_no_stride(const int* nhw_offset, tv::array<int, 2>& npq_offset)  const {
    
    npq_offset = nhw_to_npq<true>(nhw_offset);
    return (npq_offset[0] < problem_.N) && (npq_offset[0] >= 0) && 
        npq_offset[1] >= 0 && npq_offset[1] < problem_.output_dims[0];
  }
  TV_HOST_DEVICE_INLINE bool query_nhw(const int* npq_offset, tv::array<int, 2>& nhw_offset)  const {
    
    nhw_offset = npq_to_nhw(npq_offset);
    return (nhw_offset[0] < problem_.N) && (nhw_offset[0] >= 0) && 
        nhw_offset[1] >= 0 && nhw_offset[1] < problem_.input_dims[0];
  }
  TV_HOST_DEVICE_INLINE bool query_nhw_out(const int* npq_offset, tv::array<int, 2>& nhw_offset)  const {
    
    nhw_offset = npq_to_nhw(npq_offset);
    return (nhw_offset[0] < problem_.N) && (nhw_offset[0] >= 0) && 
        nhw_offset[1] >= 0 && nhw_offset[1] < problem_.output_dims[0];
  }
};
} // namespace spinds
} // namespace ops_cpu1d
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib