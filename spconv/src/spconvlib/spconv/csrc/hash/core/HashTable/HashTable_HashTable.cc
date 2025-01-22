#include <spconvlib/spconv/csrc/hash/core/HashTable.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace hash {
namespace core {
using TensorView = spconvlib::cumm::common::TensorView;
using TslRobinMap = spconvlib::cumm::common::TslRobinMap;
 HashTable::HashTable(bool is_cpu, int key_itemsize, int value_itemsize, tv::Tensor keys_data, tv::Tensor values_data, std::uintptr_t stream) : is_cpu(is_cpu), keys_data(keys_data), values_data(values_data), key_itemsize_(key_itemsize), value_itemsize_(value_itemsize), insert_count_(0)  {
  
  TV_ASSERT_RT_ERR(key_itemsize == 4 || key_itemsize == 8, "key_itemsize must be 4 or 8");
  TV_ASSERT_RT_ERR(value_itemsize == 4 || value_itemsize == 8, "value_itemsize must be 4 or 8");
  if (!is_cpu){
      TV_ASSERT_RT_ERR(!keys_data.empty() && !values_data.empty(), "key and value must not empty");
      TV_ASSERT_RT_ERR(keys_data.dim(0) == values_data.dim(0), "key and value must have same size");
      TV_ASSERT_RT_ERR(key_itemsize == keys_data.itemsize(), "key_itemsize must equal to key_data");
      TV_ASSERT_RT_ERR(value_itemsize == values_data.itemsize(), "value_itemsize must equal to values_data");
      // clear cuda table here.
      clear(stream);
  }
}
} // namespace core
} // namespace hash
} // namespace csrc
} // namespace spconv
} // namespace spconvlib