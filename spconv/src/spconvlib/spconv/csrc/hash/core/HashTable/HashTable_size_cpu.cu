#include <spconvlib/spconv/csrc/hash/core/HashTable.h>
namespace spconvlib {
namespace spconv {
namespace csrc {
namespace hash {
namespace core {
using TensorView = spconvlib::cumm::common::TensorView;
using TslRobinMap = spconvlib::cumm::common::TslRobinMap;
int64_t HashTable::size_cpu()   {
  
  int64_t res = -1;
  TV_ASSERT_RT_ERR(is_cpu, "size_cpu can only be used in cpu hash table");
  if (is_cpu){
    {
      bool found = false;
      if (key_itemsize_ == 4 && value_itemsize_ == 4){
        auto& cpu_map = map_4_4;
        res = cpu_map.size();
        found = true;
      }
      if (key_itemsize_ == 4 && value_itemsize_ == 8){
        auto& cpu_map = map_4_8;
        res = cpu_map.size();
        found = true;
      }
      if (key_itemsize_ == 8 && value_itemsize_ == 4){
        auto& cpu_map = map_8_4;
        res = cpu_map.size();
        found = true;
      }
      if (key_itemsize_ == 8 && value_itemsize_ == 8){
        auto& cpu_map = map_8_8;
        res = cpu_map.size();
        found = true;
      }
      TV_ASSERT_RT_ERR(found, "suitable hash table not found.");
    }
  }
  return res;
}
} // namespace core
} // namespace hash
} // namespace csrc
} // namespace spconv
} // namespace spconvlib