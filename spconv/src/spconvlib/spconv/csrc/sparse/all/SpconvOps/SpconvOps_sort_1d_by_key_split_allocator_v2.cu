#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/spconv/csrc/sparse/all/CustomThrustLib.h>
#include <spconvlib/cumm/common/TensorViewKernel.h>
#include <spconvlib/spconv/csrc/sparse/all/cudakers/CudaCommonKernel.h>
#include <cub/cub.cuh>

        template <typename T> __global__ void apply_mask_kernel(const T* inp, T* out, T mask, int size){
            for (int i : tv::KernelLoopX<int>(size)){
                out[i] = inp[i] & mask;
            }
        }
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

namespace {

constexpr size_t kAlignment = 256;

inline size_t align_up(size_t n, size_t alignment) {
  return (n + alignment - 1) & ~(alignment - 1);
}

template <typename KeyT>
void cub_sort_pairs_masked(KeyT* keys, int32_t* values, int num_items,
                           KeyT mask_val, bool do_mask_output,
                           ThrustAllocator& allocator, cudaStream_t stream) {
  // Query temp storage size
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
      nullptr, temp_bytes,
      static_cast<const KeyT*>(nullptr), static_cast<KeyT*>(nullptr),
      static_cast<const int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
      num_items, 0, sizeof(KeyT) * 8, stream);

  // Allocate workspace: temp_storage + masked_keys + alt_keys + alt_values (aligned)
  size_t temp_aligned = align_up(temp_bytes, kAlignment);
  size_t key_bytes = static_cast<size_t>(num_items) * sizeof(KeyT);
  size_t key_aligned = align_up(key_bytes, kAlignment);
  size_t val_bytes = static_cast<size_t>(num_items) * sizeof(int32_t);
  size_t total_bytes = temp_aligned + key_aligned + key_aligned + val_bytes;
  char* workspace = allocator.allocate(total_bytes);

  void* d_temp = workspace;
  KeyT* masked_keys = reinterpret_cast<KeyT*>(workspace + temp_aligned);
  KeyT* alt_keys = reinterpret_cast<KeyT*>(workspace + temp_aligned + key_aligned);
  int32_t* alt_values = reinterpret_cast<int32_t*>(
      workspace + temp_aligned + key_aligned + key_aligned);

  // Apply mask to create sort keys
  tv::cuda::Launch launcher(num_items, stream);
  launcher(apply_mask_kernel<KeyT>, keys, masked_keys, mask_val, num_items);

  // Sort masked keys with values (indices)
  cub::DoubleBuffer<KeyT> d_keys(masked_keys, alt_keys);
  cub::DoubleBuffer<int32_t> d_values(values, alt_values);

  cub::DeviceRadixSort::SortPairs(
      d_temp, temp_bytes, d_keys, d_values,
      num_items, 0, sizeof(KeyT) * 8, stream);

  // Copy values back if result ended up in alt buffer
  if (d_values.Current() != values) {
    cudaMemcpyAsync(values, d_values.Current(),
        num_items * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
  }

  allocator.deallocate(workspace, total_bytes);

  // Optionally mask the original data
  if (do_mask_output) {
    launcher(mask_input<KeyT>, keys, mask_val, num_items);
  }
}

}  // namespace

tv::Tensor SpconvOps::sort_1d_by_key_split_allocator_v2(tv::Tensor data, ThrustAllocator& allocator, tv::Tensor mask, tv::Tensor indices, std::uintptr_t stream, bool mask_output)   {

  cudaStream_t stream_cu = reinterpret_cast<cudaStream_t>(stream);
  if (indices.empty()){
      indices = tv::empty({data.dim(0)}, tv::int32, 0);
  }
  tv::cuda::Launch launcher(data.dim(0), stream_cu);
  launcher(cudakers::arange_kernel<int32_t>, indices.data_ptr<int32_t>(), indices.dim(0));

  int num_items = data.dim(0);

  tv::dispatch<int32_t, uint32_t, int64_t, uint64_t>(data.dtype(), [&](auto I){
      using T = TV_DECLTYPE(I);
      auto masks_ptr = mask.data_ptr<T>();
      T mask_val = masks_ptr[0];
      cub_sort_pairs_masked(data.data_ptr<T>(), indices.data_ptr<int32_t>(),
                            num_items, mask_val, mask_output,
                            allocator, stream_cu);
  });

  return indices;
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib
