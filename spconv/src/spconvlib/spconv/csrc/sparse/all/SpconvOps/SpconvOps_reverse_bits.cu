#include <spconvlib/spconv/csrc/sparse/all/SpconvOps.h>
#include <spconvlib/cumm/common/TensorViewKernel.h>

        __global__ void reverse_bits_kernel_64(const uint64_t* data, uint64_t* out, int size){
            for (int i : tv::KernelLoopX<int>(size)){
                out[i] = __brevll(reinterpret_cast<const unsigned long long*>(data)[i]);
            }
        }

        __global__ void reverse_bits_kernel(const uint32_t* data, uint32_t* out, int size){
            for (int i : tv::KernelLoopX<int>(size)){
                out[i] = __brev(data[i]);
            }
        }

        uint32_t reverse(uint32_t x)
        {
            x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
            x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
            x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
            x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
            x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);
            return x;
        }

        int reverse(uint64_t i)
        {
            return (reverse(uint32_t(i)) << 32) | reverse(uint32_t(i >> 32));
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
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
tv::Tensor SpconvOps::reverse_bits(tv::Tensor a)   {
  
  tv::Tensor res(a.shape(), a.dtype(), a.device());
  tv::dispatch<uint32_t, uint64_t>(a.dtype(), [&](auto I){
      using T = TV_DECLTYPE(I);
      auto res_ptr = res.data_ptr<T>();
      auto a_ptr = a.data_ptr<const T>();
      if (a.device() == -1){
          for (int i = 0; i < a.size(); ++i){
              res_ptr[i] = reverse(a_ptr[i]);
          }
      }else{
          tv::cuda::Launch launcher(a.size());
          tv::if_constexpr<std::is_same<T, uint64_t>::value>([=](auto _)mutable{
              launcher(_(reverse_bits_kernel_64), a_ptr, res_ptr, int(a.size()));
          }, [=](auto _)mutable{
              launcher(_(reverse_bits_kernel), a_ptr, res_ptr, int(a.size()));
          });
      }
  });
  return res;
}
} // namespace all
} // namespace sparse
} // namespace csrc
} // namespace spconv
} // namespace spconvlib