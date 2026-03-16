#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/all/CustomThrustLib.h>
#include <spconvlib/cumm/common/TensorViewKernel.h>
#include <spconvlib/spconv/csrc/sparse/all/cudakers/CudaCommonKernel.h>
#include <cub/cub.cuh>
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

namespace {

constexpr size_t kAlignment = 256;

inline size_t align_up(size_t n, size_t alignment) {
  return (n + alignment - 1) & ~(alignment - 1);
}

template <typename KeyT>
void cub_sort_pairs(KeyT* keys, int32_t* values, int num_items,
                    ThrustAllocator& allocator, cudaStream_t stream) {
  // Query temp storage size
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr, temp_bytes,
      static_cast<const KeyT*>(nullptr), static_cast<KeyT*>(nullptr),
      static_cast<const int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
      num_items, 0, sizeof(KeyT) * 8, stream);

  // Allocate workspace: temp_storage + alt_keys + alt_values (aligned)
  size_t temp_aligned = align_up(temp_bytes, kAlignment);
  size_t alt_keys_bytes = static_cast<size_t>(num_items) * sizeof(KeyT);
  size_t alt_keys_aligned = align_up(alt_keys_bytes, kAlignment);
  size_t alt_values_bytes = static_cast<size_t>(num_items) * sizeof(int32_t);
  size_t total_bytes = temp_aligned + alt_keys_aligned + alt_values_bytes;
  char* workspace = allocator.allocate(total_bytes);

  void* d_temp = workspace;
  KeyT* alt_keys = reinterpret_cast<KeyT*>(workspace + temp_aligned);
  int32_t* alt_values = reinterpret_cast<int32_t*>(
      workspace + temp_aligned + alt_keys_aligned);

  cub::DoubleBuffer<KeyT> d_keys(keys, alt_keys);
  cub::DoubleBuffer<int32_t> d_values(values, alt_values);

  cub::DeviceRadixSort::SortPairs(
      d_temp, temp_bytes, d_keys, d_values,
      num_items, 0, sizeof(KeyT) * 8, stream);

  // Copy back if result ended up in alt buffer
  if (d_keys.Current() != keys) {
    cudaMemcpyAsync(keys, d_keys.Current(),
        num_items * sizeof(KeyT), cudaMemcpyDeviceToDevice, stream);
  }
  if (d_values.Current() != values) {
    cudaMemcpyAsync(values, d_values.Current(),
        num_items * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
  }

  allocator.deallocate(workspace, total_bytes);
}

}  // namespace

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

  int num_items = data.dim(0);

  if (mask_count == 1) {
    // Use CUB radix sort (CUDA graph capture compatible)
    if (data.dtype() == tv::DType(1)){
      cub_sort_pairs(data.data_ptr<int32_t>(), indices.data_ptr<int32_t>(),
                     num_items, allocator, stream_cu);
    }
    else if (data.dtype() == tv::DType(8)){
      cub_sort_pairs(data.data_ptr<int64_t>(), indices.data_ptr<int32_t>(),
                     num_items, allocator, stream_cu);
    }
    else if (data.dtype() == tv::DType(10)){
      cub_sort_pairs(data.data_ptr<uint32_t>(), indices.data_ptr<int32_t>(),
                     num_items, allocator, stream_cu);
    }
    else if (data.dtype() == tv::DType(11)){
      cub_sort_pairs(data.data_ptr<uint64_t>(), indices.data_ptr<int32_t>(),
                     num_items, allocator, stream_cu);
    }
    else{
      TV_THROW_RT_ERR("unknown dtype data.dtype(), available: [int32_t, int64_t, uint32_t, uint64_t]")
    }
  }
  else {
    // mask_count > 1: fall back to thrust (rare case for kernel_volume > 32)
    if (data.dtype() == tv::DType(1)){
      using T_ = int32_t;
      tv::dispatch_int<1, 2, 3, 4>(mask_count, [&](auto IV){
          constexpr int I = TV_DECLTYPE(IV)::value;
          using T = tv::mp_rename<tv::mp_repeat_c<tv::mp_list<T_>, I>, thrust::tuple>;
          thrust::device_ptr<T> ptr_tr(reinterpret_cast<T*>(data.data_ptr<T_>()));
          thrust::device_ptr<int32_t> ptr_k(indices.data_ptr<int32_t>());
          auto ctx2 = thrust::cuda::par(allocator).on(stream_cu);
          thrust::sort_by_key(ctx2, ptr_tr, ptr_tr + data.dim(0), ptr_k);
      });
    }
    else if (data.dtype() == tv::DType(8)){
      using T_ = int64_t;
      tv::dispatch_int<1, 2, 3, 4>(mask_count, [&](auto IV){
          constexpr int I = TV_DECLTYPE(IV)::value;
          using T = tv::mp_rename<tv::mp_repeat_c<tv::mp_list<T_>, I>, thrust::tuple>;
          thrust::device_ptr<T> ptr_tr(reinterpret_cast<T*>(data.data_ptr<T_>()));
          thrust::device_ptr<int32_t> ptr_k(indices.data_ptr<int32_t>());
          auto ctx2 = thrust::cuda::par(allocator).on(stream_cu);
          thrust::sort_by_key(ctx2, ptr_tr, ptr_tr + data.dim(0), ptr_k);
      });
    }
    else if (data.dtype() == tv::DType(10)){
      using T_ = uint32_t;
      tv::dispatch_int<1, 2, 3, 4>(mask_count, [&](auto IV){
          constexpr int I = TV_DECLTYPE(IV)::value;
          using T = tv::mp_rename<tv::mp_repeat_c<tv::mp_list<T_>, I>, thrust::tuple>;
          thrust::device_ptr<T> ptr_tr(reinterpret_cast<T*>(data.data_ptr<T_>()));
          thrust::device_ptr<int32_t> ptr_k(indices.data_ptr<int32_t>());
          auto ctx2 = thrust::cuda::par(allocator).on(stream_cu);
          thrust::sort_by_key(ctx2, ptr_tr, ptr_tr + data.dim(0), ptr_k);
      });
    }
    else if (data.dtype() == tv::DType(11)){
      using T_ = uint64_t;
      tv::dispatch_int<1, 2, 3, 4>(mask_count, [&](auto IV){
          constexpr int I = TV_DECLTYPE(IV)::value;
          using T = tv::mp_rename<tv::mp_repeat_c<tv::mp_list<T_>, I>, thrust::tuple>;
          thrust::device_ptr<T> ptr_tr(reinterpret_cast<T*>(data.data_ptr<T_>()));
          thrust::device_ptr<int32_t> ptr_k(indices.data_ptr<int32_t>());
          auto ctx2 = thrust::cuda::par(allocator).on(stream_cu);
          thrust::sort_by_key(ctx2, ptr_tr, ptr_tr + data.dim(0), ptr_k);
      });
    }
    else{
      TV_THROW_RT_ERR("unknown dtype data.dtype(), available: [int32_t, int64_t, uint32_t, uint64_t]")
    }
  }

  return indices;
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib
