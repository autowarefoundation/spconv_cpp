#pragma once
#include <tensorview/parallel/all.h>
#include <spconvlib/cumm/common/TensorView.h>
#include <spconvlib/cumm/common/TslRobinMap.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace hash {
namespace core {
using TensorView = spconvlib::cumm::common::TensorView;
using TslRobinMap = spconvlib::cumm::common::TslRobinMap;
struct HashTable {
  tv::Tensor keys_data;
  tv::Tensor values_data;
  int key_itemsize_;
  int value_itemsize_;
  bool is_cpu;
  tsl::robin_map<uint32_t, uint32_t> map_4_4;
  tsl::robin_map<uint32_t, uint64_t> map_4_8;
  tsl::robin_map<uint64_t, uint32_t> map_8_4;
  tsl::robin_map<uint64_t, uint64_t> map_8_8;
  int64_t insert_count_;
  /**
   * @param is_cpu 
   * @param key_itemsize 
   * @param value_itemsize 
   * @param keys_data 
   * @param values_data 
   * @param stream 
   */
   HashTable(bool is_cpu, int key_itemsize, int value_itemsize, tv::Tensor keys_data, tv::Tensor values_data, std::uintptr_t stream = 0);
  /**
   * in this function, if values is empty, it will be assigned to zero.
   *         
   * @param stream 
   */
  void clear(std::uintptr_t stream = 0);
  /**
   * in this function, if values is empty, it will be assigned to zero.
   *         
   * @param keys 
   * @param values 
   * @param stream 
   */
  void insert(tv::Tensor keys, tv::Tensor values = tv::Tensor(), std::uintptr_t stream = 0);
  /**
   * query keys, save to values, and save is_empty to is_empty
   *         
   * @param keys 
   * @param values 
   * @param is_empty 
   * @param stream 
   */
  void query(tv::Tensor keys, tv::Tensor values, tv::Tensor is_empty, std::uintptr_t stream);
  /**
   * this function assign "arange(NumItem)" to table values.
   * useful in "unique-like" operations.
   * unlike insert/query, this method only support i32/i64/u32/u64 for value.
   * count must be u32/u64.
   * @param count 
   * @param stream 
   */
  void assign_arange_(tv::Tensor count, std::uintptr_t stream = 0);
  /**
   * this function can only be used to get cpu hash table size.
   *         
   */
  int64_t size_cpu();
  /**
   * get items.
   *         
   * @param keys 
   * @param values 
   * @param count 
   * @param stream 
   */
  void items(tv::Tensor keys, tv::Tensor values, tv::Tensor count, std::uintptr_t stream);
  /**
   * insert v of given k if k exists. won't insert any new key.
   *         
   * @param keys 
   * @param values 
   * @param is_empty 
   * @param stream 
   */
  void insert_exist_keys(tv::Tensor keys, tv::Tensor values, tv::Tensor is_empty, std::uintptr_t stream);
};
} // namespace core
} // namespace hash
} // namespace csrc
} // namespace spconv
} // namespace spconvlib