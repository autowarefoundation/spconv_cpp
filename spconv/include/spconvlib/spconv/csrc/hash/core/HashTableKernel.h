#pragma once
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/cumm/common/TensorViewHashKernel.h>
#include <spconvlib/cumm/common/TensorViewKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace hash {
namespace core {
using TensorView = spconvlib::cumm::common::TensorView;
using TensorViewHashKernel = spconvlib::cumm::common::TensorViewHashKernel;
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
template <typename THashTableSplit>
__global__ void insert_exist_keys_kernel(THashTableSplit table, const typename THashTableSplit::key_type *__restrict__ key_ptr, const typename THashTableSplit::mapped_type *__restrict__ value_ptr, uint8_t* is_empty_ptr, size_t size)   {
  
  auto value_data = table.value_ptr();
  for (size_t i : tv::KernelLoopX<size_t>(size)){
      auto key = key_ptr[i];
      auto offset = table.lookup_offset(key);
      is_empty_ptr[i] = offset == -1;
      if (offset != -1){
          value_data[offset] = value_ptr[i];
      }
  }
}
struct HashTableKernel {
};
} // namespace core
} // namespace hash
} // namespace csrc
} // namespace spconv
} // namespace spconvlib