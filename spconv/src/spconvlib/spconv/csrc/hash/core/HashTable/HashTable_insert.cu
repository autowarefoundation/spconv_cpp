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
void HashTable::insert(tv::Tensor keys, tv::Tensor values, std::uintptr_t stream)   {
  
  if (!is_cpu){
      int64_t value_after_insert = keys.dim(0) + insert_count_;
      TV_ASSERT_RT_ERR(value_after_insert < keys_data.dim(0), "inserted count exceed maximum hash size");
      insert_count_ += keys.dim(0);
      TV_ASSERT_RT_ERR(keys.dtype() == keys_data.dtype(), "keys dtype not equal to", keys_data.dtype());
  }
  auto N = keys.dim(0);
  if (!values.empty()){
      TV_ASSERT_RT_ERR(values.itemsize() == value_itemsize_, "values itemsize not equal to", value_itemsize_);
      TV_ASSERT_RT_ERR(keys.dim(0) == values.dim(0), "number of key and value must same");
  }
  if (is_cpu){
    {
      bool found = false;
      if (key_itemsize_ == 4 && value_itemsize_ == 4){
        auto& cpu_map = map_4_4;
        auto k_ptr = reinterpret_cast<const uint32_t*>(keys.raw_data());
        if (values.empty()){
            for (size_t i = 0; i < N; ++i){
                cpu_map.insert({k_ptr[i], uint32_t(0)});
            }
        }
        else{
            auto v_ptr = reinterpret_cast<const uint32_t*>(values.raw_data());
            for (size_t i = 0; i < N; ++i){
                cpu_map.insert({k_ptr[i], v_ptr[i]});
            }
        }
        found = true;
      }
      if (key_itemsize_ == 4 && value_itemsize_ == 8){
        auto& cpu_map = map_4_8;
        auto k_ptr = reinterpret_cast<const uint32_t*>(keys.raw_data());
        if (values.empty()){
            for (size_t i = 0; i < N; ++i){
                cpu_map.insert({k_ptr[i], uint64_t(0)});
            }
        }
        else{
            auto v_ptr = reinterpret_cast<const uint64_t*>(values.raw_data());
            for (size_t i = 0; i < N; ++i){
                cpu_map.insert({k_ptr[i], v_ptr[i]});
            }
        }
        found = true;
      }
      if (key_itemsize_ == 8 && value_itemsize_ == 4){
        auto& cpu_map = map_8_4;
        auto k_ptr = reinterpret_cast<const uint64_t*>(keys.raw_data());
        if (values.empty()){
            for (size_t i = 0; i < N; ++i){
                cpu_map.insert({k_ptr[i], uint32_t(0)});
            }
        }
        else{
            auto v_ptr = reinterpret_cast<const uint32_t*>(values.raw_data());
            for (size_t i = 0; i < N; ++i){
                cpu_map.insert({k_ptr[i], v_ptr[i]});
            }
        }
        found = true;
      }
      if (key_itemsize_ == 8 && value_itemsize_ == 8){
        auto& cpu_map = map_8_8;
        auto k_ptr = reinterpret_cast<const uint64_t*>(keys.raw_data());
        if (values.empty()){
            for (size_t i = 0; i < N; ++i){
                cpu_map.insert({k_ptr[i], uint64_t(0)});
            }
        }
        else{
            auto v_ptr = reinterpret_cast<const uint64_t*>(values.raw_data());
            for (size_t i = 0; i < N; ++i){
                cpu_map.insert({k_ptr[i], v_ptr[i]});
            }
        }
        found = true;
      }
      TV_ASSERT_RT_ERR(found, "suitable hash table not found.");
    }
  }
  else{
    auto custream = reinterpret_cast<cudaStream_t>(stream);
    if (keys_data.dtype() == tv::DType(1)){
      using K = int32_t;
      K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
      const K* key_ptr = reinterpret_cast<const K*>(keys.raw_data());
      if (values_data.itemsize() == 4){
        using V = tv::hash::itemsize_to_unsigned_t<4>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        const V* value_ptr = reinterpret_cast<const V*>(values.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        tv::cuda::Launch launcher(N, custream);
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        launcher(tv::hash::insert_split<table_t>, table, key_ptr, value_ptr, size_t(N));
      }
      else if (values_data.itemsize() == 8){
        using V = tv::hash::itemsize_to_unsigned_t<8>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        const V* value_ptr = reinterpret_cast<const V*>(values.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        tv::cuda::Launch launcher(N, custream);
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        launcher(tv::hash::insert_split<table_t>, table, key_ptr, value_ptr, size_t(N));
      }
      else{
        TV_THROW_RT_ERR("unknown val values_data.itemsize(), available: [4, 8]")
      }
    }
    else if (keys_data.dtype() == tv::DType(8)){
      using K = int64_t;
      K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
      const K* key_ptr = reinterpret_cast<const K*>(keys.raw_data());
      if (values_data.itemsize() == 4){
        using V = tv::hash::itemsize_to_unsigned_t<4>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        const V* value_ptr = reinterpret_cast<const V*>(values.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        tv::cuda::Launch launcher(N, custream);
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        launcher(tv::hash::insert_split<table_t>, table, key_ptr, value_ptr, size_t(N));
      }
      else if (values_data.itemsize() == 8){
        using V = tv::hash::itemsize_to_unsigned_t<8>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        const V* value_ptr = reinterpret_cast<const V*>(values.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        tv::cuda::Launch launcher(N, custream);
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        launcher(tv::hash::insert_split<table_t>, table, key_ptr, value_ptr, size_t(N));
      }
      else{
        TV_THROW_RT_ERR("unknown val values_data.itemsize(), available: [4, 8]")
      }
    }
    else if (keys_data.dtype() == tv::DType(10)){
      using K = uint32_t;
      K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
      const K* key_ptr = reinterpret_cast<const K*>(keys.raw_data());
      if (values_data.itemsize() == 4){
        using V = tv::hash::itemsize_to_unsigned_t<4>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        const V* value_ptr = reinterpret_cast<const V*>(values.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        tv::cuda::Launch launcher(N, custream);
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        launcher(tv::hash::insert_split<table_t>, table, key_ptr, value_ptr, size_t(N));
      }
      else if (values_data.itemsize() == 8){
        using V = tv::hash::itemsize_to_unsigned_t<8>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        const V* value_ptr = reinterpret_cast<const V*>(values.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        tv::cuda::Launch launcher(N, custream);
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        launcher(tv::hash::insert_split<table_t>, table, key_ptr, value_ptr, size_t(N));
      }
      else{
        TV_THROW_RT_ERR("unknown val values_data.itemsize(), available: [4, 8]")
      }
    }
    else if (keys_data.dtype() == tv::DType(11)){
      using K = uint64_t;
      K* key_data_ptr = reinterpret_cast<K*>(keys_data.raw_data());
      const K* key_ptr = reinterpret_cast<const K*>(keys.raw_data());
      if (values_data.itemsize() == 4){
        using V = tv::hash::itemsize_to_unsigned_t<4>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        const V* value_ptr = reinterpret_cast<const V*>(values.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        tv::cuda::Launch launcher(N, custream);
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        launcher(tv::hash::insert_split<table_t>, table, key_ptr, value_ptr, size_t(N));
      }
      else if (values_data.itemsize() == 8){
        using V = tv::hash::itemsize_to_unsigned_t<8>;
        V* value_data_ptr = reinterpret_cast<V*>(values_data.raw_data());
        const V* value_ptr = reinterpret_cast<const V*>(values.raw_data());
        using table_t =
            tv::hash::LinearHashTableSplit<K, V, tv::hash::Murmur3Hash<K>,
                                        tv::hash::default_empty_key_v<K>, false>;
        tv::cuda::Launch launcher(N, custream);
        table_t table(key_data_ptr, value_data_ptr, keys_data.dim(0));
        launcher(tv::hash::insert_split<table_t>, table, key_ptr, value_ptr, size_t(N));
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