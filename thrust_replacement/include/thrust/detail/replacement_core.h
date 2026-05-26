#pragma once

#include <cstddef>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace thrust {

template <typename T>
class device_ptr {
public:
  using element_type = T;
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = T *;
  using reference = T &;
  using iterator_category = std::random_access_iterator_tag;

  constexpr device_ptr() = default;
  constexpr explicit device_ptr(T *ptr) : ptr_(ptr) {}

  constexpr T *get() const { return ptr_; }
  constexpr explicit operator bool() const { return ptr_ != nullptr; }
  constexpr operator T *() const { return ptr_; }

  constexpr reference operator*() const { return *ptr_; }
  constexpr reference operator[](difference_type n) const { return ptr_[n]; }

  constexpr device_ptr &operator++() {
    ++ptr_;
    return *this;
  }

  constexpr device_ptr operator++(int) {
    device_ptr tmp(*this);
    ++(*this);
    return tmp;
  }

  constexpr device_ptr &operator--() {
    --ptr_;
    return *this;
  }

  constexpr device_ptr operator--(int) {
    device_ptr tmp(*this);
    --(*this);
    return tmp;
  }

  constexpr device_ptr &operator+=(difference_type n) {
    ptr_ += n;
    return *this;
  }

  constexpr device_ptr &operator-=(difference_type n) {
    ptr_ -= n;
    return *this;
  }

  friend constexpr device_ptr operator+(device_ptr ptr, difference_type n) {
    ptr += n;
    return ptr;
  }

  friend constexpr device_ptr operator+(difference_type n, device_ptr ptr) {
    ptr += n;
    return ptr;
  }

  friend constexpr device_ptr operator-(device_ptr ptr, difference_type n) {
    ptr -= n;
    return ptr;
  }

  friend constexpr difference_type operator-(device_ptr lhs, device_ptr rhs) {
    return lhs.ptr_ - rhs.ptr_;
  }

  friend constexpr bool operator==(device_ptr lhs, device_ptr rhs) {
    return lhs.ptr_ == rhs.ptr_;
  }

  friend constexpr bool operator!=(device_ptr lhs, device_ptr rhs) {
    return !(lhs == rhs);
  }

  friend constexpr bool operator<(device_ptr lhs, device_ptr rhs) {
    return lhs.ptr_ < rhs.ptr_;
  }

private:
  T *ptr_ = nullptr;
};

template <typename... Ts>
using tuple = std::tuple<Ts...>;

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
                   InputIt2 first2, OutputIt result, BinaryOperation op) {
  for (; first != last; ++first, ++first2, ++result) {
    *result = op(*first, *first2);
  }
  return result;
}

template <typename Policy, typename InputIt, typename OutputIt,
          typename UnaryOperation>
OutputIt transform(const Policy &, InputIt first, InputIt last, OutputIt result,
                   UnaryOperation op) {
  for (; first != last; ++first, ++result) {
    *result = op(*first);
  }
  return result;
}

template <typename Policy, typename InputIt, typename T,
          typename BinaryOperation>
T reduce(const Policy &, InputIt first, InputIt last, T init,
         BinaryOperation op) {
  for (; first != last; ++first) {
    init = op(init, *first);
  }
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

} // namespace thrust
