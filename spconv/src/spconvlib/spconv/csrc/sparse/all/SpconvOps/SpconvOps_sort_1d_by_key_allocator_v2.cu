#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/all/CustomThrustLib.h>
#include <spconvlib/cumm/common/TensorViewKernel.h>
#include <spconvlib/spconv/csrc/sparse/all/cudakers/CudaCommonKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace sparse {
namespace all {
using ThrustCustomAllocatorV2 = spconvlib::spconv::csrc::sparse::all::ThrustCustomAllocatorV2;
using ExternalAllocator = spconvlib::spconv::csrc::sparse::alloc::ExternalAllocator;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using ThrustAllocator = spconvlib::spconv::csrc::sparse::alloc::ThrustAllocator;
using Point2Voxel1DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu1d::Point2VoxelCPU;
using SpconvIndicesCPU1D = spconvlib::spconv::csrc::sparse::all::ops_cpu1d::SparseConvIndicesCPU;
using Point2Voxel1D = spconvlib::spconv::csrc::sparse::all::ops1d::Point2Voxel;
using Point2Voxel2DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu2d::Point2VoxelCPU;
using SpconvIndicesCPU2D = spconvlib::spconv::csrc::sparse::all::ops_cpu2d::SparseConvIndicesCPU;
using Point2Voxel2D = spconvlib::spconv::csrc::sparse::all::ops2d::Point2Voxel;
using Point2Voxel3DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu3d::Point2VoxelCPU;
using SpconvIndicesCPU3D = spconvlib::spconv::csrc::sparse::all::ops_cpu3d::SparseConvIndicesCPU;
using Point2Voxel3D = spconvlib::spconv::csrc::sparse::all::ops3d::Point2Voxel;
using Point2Voxel4DCPU = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::Point2VoxelCPU;
using SpconvIndicesCPU4D = spconvlib::spconv::csrc::sparse::all::ops_cpu4d::SparseConvIndicesCPU;
using Point2Voxel4D = spconvlib::spconv::csrc::sparse::all::ops4d::Point2Voxel;
using CustomThrustLib = spconvlib::spconv::csrc::sparse::all::CustomThrustLib;
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
tv::Tensor SpconvOps::sort_1d_by_key_allocator_v2(tv::Tensor data, ThrustAllocator& allocator, tv::Tensor indices, std::uintptr_t stream, int mask_count, bool do_sort)   {
  
  cudaStream_t stream_cu = reinterpret_cast<cudaStream_t>(stream);
  if (indices.empty()){
      indices = tv::empty({data.dim(0)}, tv::int32, 0);
  }
  tv::cuda::Launch launcher(data.dim(0), stream_cu);
  launcher(cudakers::arange_kernel<int32_t>, indices.data_ptr<int32_t>(), indices.dim(0));
  if (!do_sort){
      return indices;
  }
  // auto timer = tv::CUDATimer();
  if (data.dtype() == tv::DType(1)){
    using T_ = int32_t;
    tv::dispatch_int<1, 2, 3, 4>(mask_count, [&](auto IV){
        constexpr int I = TV_DECLTYPE(IV)::value;
        // we can't use thrust::tuple in mp_repeat_c directly because
        // thrust tuple actually has fixed size template arguments.
        using T = tv::mp_rename<tv::mp_repeat_c<tv::mp_list<T_>, I>, thrust::tuple>;
        thrust::device_ptr<T> ptr_tr(reinterpret_cast<T*>(data.data_ptr<T_>()));
        thrust::device_ptr<int32_t> ptr_k(indices.data_ptr<int32_t>());
        auto thrust_ctx = thrust::cuda::par.on(stream_cu);
        auto ctx2 = thrust::cuda::par(allocator).on(stream_cu);
        thrust::sort_by_key(ctx2, ptr_tr, ptr_tr + data.dim(0), ptr_k);
    });
  }
  else if (data.dtype() == tv::DType(8)){
    using T_ = int64_t;
    tv::dispatch_int<1, 2, 3, 4>(mask_count, [&](auto IV){
        constexpr int I = TV_DECLTYPE(IV)::value;
        // we can't use thrust::tuple in mp_repeat_c directly because
        // thrust tuple actually has fixed size template arguments.
        using T = tv::mp_rename<tv::mp_repeat_c<tv::mp_list<T_>, I>, thrust::tuple>;
        thrust::device_ptr<T> ptr_tr(reinterpret_cast<T*>(data.data_ptr<T_>()));
        thrust::device_ptr<int32_t> ptr_k(indices.data_ptr<int32_t>());
        auto thrust_ctx = thrust::cuda::par.on(stream_cu);
        auto ctx2 = thrust::cuda::par(allocator).on(stream_cu);
        thrust::sort_by_key(ctx2, ptr_tr, ptr_tr + data.dim(0), ptr_k);
    });
  }
  else if (data.dtype() == tv::DType(10)){
    using T_ = uint32_t;
    tv::dispatch_int<1, 2, 3, 4>(mask_count, [&](auto IV){
        constexpr int I = TV_DECLTYPE(IV)::value;
        // we can't use thrust::tuple in mp_repeat_c directly because
        // thrust tuple actually has fixed size template arguments.
        using T = tv::mp_rename<tv::mp_repeat_c<tv::mp_list<T_>, I>, thrust::tuple>;
        thrust::device_ptr<T> ptr_tr(reinterpret_cast<T*>(data.data_ptr<T_>()));
        thrust::device_ptr<int32_t> ptr_k(indices.data_ptr<int32_t>());
        auto thrust_ctx = thrust::cuda::par.on(stream_cu);
        auto ctx2 = thrust::cuda::par(allocator).on(stream_cu);
        thrust::sort_by_key(ctx2, ptr_tr, ptr_tr + data.dim(0), ptr_k);
    });
  }
  else if (data.dtype() == tv::DType(11)){
    using T_ = uint64_t;
    tv::dispatch_int<1, 2, 3, 4>(mask_count, [&](auto IV){
        constexpr int I = TV_DECLTYPE(IV)::value;
        // we can't use thrust::tuple in mp_repeat_c directly because
        // thrust tuple actually has fixed size template arguments.
        using T = tv::mp_rename<tv::mp_repeat_c<tv::mp_list<T_>, I>, thrust::tuple>;
        thrust::device_ptr<T> ptr_tr(reinterpret_cast<T*>(data.data_ptr<T_>()));
        thrust::device_ptr<int32_t> ptr_k(indices.data_ptr<int32_t>());
        auto thrust_ctx = thrust::cuda::par.on(stream_cu);
        auto ctx2 = thrust::cuda::par(allocator).on(stream_cu);
        thrust::sort_by_key(ctx2, ptr_tr, ptr_tr + data.dim(0), ptr_k);
    });
  }
  else{
    TV_THROW_RT_ERR("unknown dtype data.dtype(), available: [int32_t, int64_t, uint32_t, uint64_t]")
  }
  // tv::ssprint("SORT BY KEY TIME", data.dim(0), timer.report() / 1000.0);
  return indices;
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib