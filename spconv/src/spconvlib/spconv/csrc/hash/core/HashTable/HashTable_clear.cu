#include <spconvlib/spconv/csrc/hash/core/HashTable.h>
#include <spconvlib/cumm/common/TensorViewHashKernel.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace hash {
namespace core {
using TensorView = spconvlib::cumm::common::TensorView;
using TslRobinMap = spconvlib::cumm::common::TslRobinMap;
using TensorViewHashKernel = spconvlib::cumm::common::TensorViewHashKernel;
void HashTable::clear(std::uintptr_t stream)   {
  
  if (is_cpu){
    if (is_cpu){
        map_4_4.clear();
        map_4_8.clear();
        map_8_4.clear();
        map_8_8.clear();
        return;
    }
  }
  else{
    auto custream = reinterpret_cast<cudaStream_t>(stream);
    if (keys_data.dtype() == tv::DType(1)){
      using K = int32_t;
      K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
      if (values_data.itemsize() == 4){
        using V = tv::hash::itemsize_to_unsigned_t<4>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::clear_map_kernel_split<table_t>, table);
      }
      else if (values_data.itemsize() == 8){
        using V = tv::hash::itemsize_to_unsigned_t<8>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::clear_map_kernel_split<table_t>, table);
      }
      else{
        TV_THROW_RT_ERR("unknown val values_data.itemsize(), available: [4, 8]")
      }
    }
    else if (keys_data.dtype() == tv::DType(8)){
      using K = int64_t;
      K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
      if (values_data.itemsize() == 4){
        using V = tv::hash::itemsize_to_unsigned_t<4>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::clear_map_kernel_split<table_t>, table);
      }
      else if (values_data.itemsize() == 8){
        using V = tv::hash::itemsize_to_unsigned_t<8>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::clear_map_kernel_split<table_t>, table);
      }
      else{
        TV_THROW_RT_ERR("unknown val values_data.itemsize(), available: [4, 8]")
      }
    }
    else if (keys_data.dtype() == tv::DType(10)){
      using K = uint32_t;
      K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
      if (values_data.itemsize() == 4){
        using V = tv::hash::itemsize_to_unsigned_t<4>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::clear_map_kernel_split<table_t>, table);
      }
      else if (values_data.itemsize() == 8){
        using V = tv::hash::itemsize_to_unsigned_t<8>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::clear_map_kernel_split<table_t>, table);
      }
      else{
        TV_THROW_RT_ERR("unknown val values_data.itemsize(), available: [4, 8]")
      }
    }
    else if (keys_data.dtype() == tv::DType(11)){
      using K = uint64_t;
      K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
      if (values_data.itemsize() == 4){
        using V = tv::hash::itemsize_to_unsigned_t<4>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::clear_map_kernel_split<table_t>, table);
      }
      else if (values_data.itemsize() == 8){
        using V = tv::hash::itemsize_to_unsigned_t<8>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::clear_map_kernel_split<table_t>, table);
      }
      else{
        TV_THROW_RT_ERR("unknown val values_data.itemsize(), available: [4, 8]")
      }
    }
    else{
      TV_THROW_RT_ERR("unknown dtype keys_data.dtype(), available: [int32_t, int64_t, uint32_t, uint64_t]")
    }
  }
}
} // namespace core
} // namespace hash
} // namespace csrc
} // namespace spconv
} // namespace spconvlib