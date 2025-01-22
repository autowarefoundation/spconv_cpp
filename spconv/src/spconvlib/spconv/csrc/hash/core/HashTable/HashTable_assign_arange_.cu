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
void HashTable::assign_arange_(tv::Tensor count, std::uintptr_t stream)   {
  
  if (is_cpu){
    {
      bool found = false;
      if (key_itemsize_ == 4 && value_itemsize_ == 4){
        auto& cpu_map = map_4_4;
        uint32_t index = 0;
        for (auto it = cpu_map.begin(); it != cpu_map.end(); ++it){
            it.value() = index;
            ++index;
        }
        found = true;
      }
      if (key_itemsize_ == 4 && value_itemsize_ == 8){
        auto& cpu_map = map_4_8;
        uint64_t index = 0;
        for (auto it = cpu_map.begin(); it != cpu_map.end(); ++it){
            it.value() = index;
            ++index;
        }
        found = true;
      }
      if (key_itemsize_ == 8 && value_itemsize_ == 4){
        auto& cpu_map = map_8_4;
        uint32_t index = 0;
        for (auto it = cpu_map.begin(); it != cpu_map.end(); ++it){
            it.value() = index;
            ++index;
        }
        found = true;
      }
      if (key_itemsize_ == 8 && value_itemsize_ == 8){
        auto& cpu_map = map_8_8;
        uint64_t index = 0;
        for (auto it = cpu_map.begin(); it != cpu_map.end(); ++it){
            it.value() = index;
            ++index;
        }
        found = true;
      }
      TV_ASSERT_RT_ERR(found, "suitable hash table not found.");
    }
  }
  else{
    TV_ASSERT_RT_ERR(count.device() == 0, "count must be cuda");
    auto custream = reinterpret_cast<cudaStream_t>(stream);
    if (keys_data.dtype() == tv::DType(1)){
      using K = int32_t;
      using Kunsigned = tv::hash::itemsize_to_unsigned_t<sizeof(K)>;
      auto count_ptr = count.data_ptr<Kunsigned>();
      K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
      if (values_data.dtype() == tv::DType(1)){
        using V = int32_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else if (values_data.dtype() == tv::DType(8)){
        using V = int64_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else if (values_data.dtype() == tv::DType(10)){
        using V = uint32_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else if (values_data.dtype() == tv::DType(11)){
        using V = uint64_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else{
        TV_THROW_RT_ERR("unknown dtype values_data.dtype(), available: [int32_t, int64_t, uint32_t, uint64_t]")
      }
    }
    else if (keys_data.dtype() == tv::DType(8)){
      using K = int64_t;
      using Kunsigned = tv::hash::itemsize_to_unsigned_t<sizeof(K)>;
      auto count_ptr = count.data_ptr<Kunsigned>();
      K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
      if (values_data.dtype() == tv::DType(1)){
        using V = int32_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else if (values_data.dtype() == tv::DType(8)){
        using V = int64_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else if (values_data.dtype() == tv::DType(10)){
        using V = uint32_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else if (values_data.dtype() == tv::DType(11)){
        using V = uint64_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else{
        TV_THROW_RT_ERR("unknown dtype values_data.dtype(), available: [int32_t, int64_t, uint32_t, uint64_t]")
      }
    }
    else if (keys_data.dtype() == tv::DType(10)){
      using K = uint32_t;
      using Kunsigned = tv::hash::itemsize_to_unsigned_t<sizeof(K)>;
      auto count_ptr = count.data_ptr<Kunsigned>();
      K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
      if (values_data.dtype() == tv::DType(1)){
        using V = int32_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else if (values_data.dtype() == tv::DType(8)){
        using V = int64_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else if (values_data.dtype() == tv::DType(10)){
        using V = uint32_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else if (values_data.dtype() == tv::DType(11)){
        using V = uint64_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else{
        TV_THROW_RT_ERR("unknown dtype values_data.dtype(), available: [int32_t, int64_t, uint32_t, uint64_t]")
      }
    }
    else if (keys_data.dtype() == tv::DType(11)){
      using K = uint64_t;
      using Kunsigned = tv::hash::itemsize_to_unsigned_t<sizeof(K)>;
      auto count_ptr = count.data_ptr<Kunsigned>();
      K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
      if (values_data.dtype() == tv::DType(1)){
        using V = int32_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else if (values_data.dtype() == tv::DType(8)){
        using V = int64_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else if (values_data.dtype() == tv::DType(10)){
        using V = uint32_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else if (values_data.dtype() == tv::DType(11)){
        using V = uint64_t;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        tv::cuda::Launch launcher(table.size(), custream);
        launcher(tv::hash::assign_arange_split<table_t, Kunsigned>, table, count_ptr);
      }
      else{
        TV_THROW_RT_ERR("unknown dtype values_data.dtype(), available: [int32_t, int64_t, uint32_t, uint64_t]")
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