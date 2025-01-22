#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/all/CustomThrustLib.h>
#include <spconvlib/cumm/common/TensorViewKernel.h>
#include <spconvlib/spconv/csrc/sparse/all/cudakers/CudaCommonKernel.h>

        template <typename T> struct MaskedElementComp {
            T mask_;
            TV_HOST_DEVICE_INLINE T operator()(const T &x, const T &y) const {
                return (x & mask_) < (y & mask_);
            }
        };
        template <typename T> __global__ void mask_input(T* inp, T mask, int size){
            for (int i : tv::KernelLoopX<int>(size)){
                inp[i] &= mask;
            }
        }
        
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
tv::Tensor SpconvOps::sort_1d_by_key_split_allocator_v2(tv::Tensor data, ThrustAllocator& allocator, tv::Tensor mask, tv::Tensor indices, std::uintptr_t stream, bool mask_output)   {
  
  cudaStream_t stream_cu = reinterpret_cast<cudaStream_t>(stream);
  // auto timer = tv::CudaContextTimer<>();
  if (indices.empty()){
      indices = tv::empty({data.dim(0)}, tv::int32, 0);
  }
  tv::cuda::Launch launcher(data.dim(0), stream_cu);
  launcher(cudakers::arange_kernel<int32_t>, indices.data_ptr<int32_t>(), indices.dim(0));
  tv::dispatch<int32_t, uint32_t, int64_t, uint64_t>(data.dtype(), [&](auto I){
      using T = TV_DECLTYPE(I);
      auto masks_ptr = mask.data_ptr<T>();
      MaskedElementComp<T> op_comp{masks_ptr[0]};
      thrust::device_ptr<T> ptr_tr(data.data_ptr<T>());
      thrust::device_ptr<int32_t> ptr_k(indices.data_ptr<int32_t>());
      // auto thrust_ctx = thrust::cuda::par.on(stream_cu);
      auto ctx2 = thrust::cuda::par(allocator).on(stream_cu);
      thrust::sort_by_key(ctx2, ptr_tr, ptr_tr + data.dim(0), ptr_k, op_comp);
      if (mask_output){
          launcher(mask_input<T>, data.data_ptr<T>(), masks_ptr[0], data.dim(0));
      }
  });
  // tv::ssprint("SORT_BY_KEY_MASKED", timer.report() / 1000.0);
  return indices;
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib