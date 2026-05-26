#pragma once

#include <cstddef>
#include <iterator>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <type_traits>
#include <utility>

namespace thrust_replacement {

template <typename T>
using device_ptr = ::thrust::device_ptr<T>;

template <typename... Ts>
using tuple = ::thrust::tuple<Ts...>;

template <typename T>
class device_vector;

namespace cuda {

template <typename Allocator = void>
struct execution_policy {
  using allocator_type = Allocator;

  constexpr execution_policy() = default;
  constexpr explicit execution_policy(Allocator *allocator)
      : allocator_(allocator) {}

  template <typename Stream>
  constexpr execution_policy on(Stream stream) const {
    auto copy = *this;
    copy.stream_ = reinterpret_cast<void *>(stream);
    return copy;
  }

  Allocator *allocator_ = nullptr;
  void *stream_ = nullptr;
};

template <>
struct execution_policy<void> {
  template <typename Stream>
  constexpr execution_policy on(Stream stream) const {
    auto copy = *this;
    copy.stream_ = reinterpret_cast<void *>(stream);
    return copy;
  }

  void *stream_ = nullptr;
};

struct par_t {
  template <typename Stream>
  constexpr execution_policy<> on(Stream stream) const {
    return execution_policy<>{}.on(stream);
  }

  template <typename Allocator>
  constexpr execution_policy<Allocator> operator()(Allocator &allocator) const {
    return execution_policy<Allocator>(&allocator);
  }
};

static constexpr par_t par{};

} // namespace cuda

template <typename Policy, typename InputIt1, typename InputIt2,
          typename OutputIt, typename BinaryOperation>
OutputIt transform(const Policy &, InputIt1 first, InputIt1 last,
                   InputIt2, OutputIt result, BinaryOperation) {
  return result + (last - first);
}

template <typename Policy, typename InputIt, typename OutputIt,
          typename UnaryOperation>
OutputIt transform(const Policy &, InputIt first, InputIt last, OutputIt result,
                   UnaryOperation) {
  return result + (last - first);
}

template <typename Policy, typename InputIt, typename T,
          typename BinaryOperation>
T reduce(const Policy &, InputIt, InputIt, T init, BinaryOperation) {
  return init;
}

template <typename Policy, typename RandomIt>
void sort(const Policy &, RandomIt, RandomIt) {}

template <typename Policy, typename RandomIt, typename Compare>
void sort(const Policy &, RandomIt, RandomIt, Compare) {}

template <typename Policy, typename ForwardIt>
ForwardIt unique(const Policy &, ForwardIt, ForwardIt last) {
  return last;
}

template <typename Policy, typename KeyIt, typename ValueIt>
void sort_by_key(const Policy &, KeyIt, KeyIt, ValueIt) {}

template <typename Policy, typename KeyIt, typename ValueIt, typename Compare>
void sort_by_key(const Policy &, KeyIt, KeyIt, ValueIt, Compare) {}

template <typename Policy, typename KeyIt, typename ValueIt>
void stable_sort_by_key(const Policy &, KeyIt, KeyIt, ValueIt) {}

template <typename Policy, typename KeyIt, typename ValueIt, typename Compare>
void stable_sort_by_key(const Policy &, KeyIt, KeyIt, ValueIt, Compare) {}

} // namespace thrust_replacement
