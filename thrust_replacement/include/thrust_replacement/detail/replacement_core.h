#pragma once

#include <cuda_runtime_api.h>

#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif

#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <thrust/device_ptr.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/tuple.h>
#include <type_traits>
#include <utility>

#ifndef THRUST_REPLACEMENT_HOST_DEVICE
#ifdef __CUDACC__
#define THRUST_REPLACEMENT_HOST_DEVICE __host__ __device__
#define THRUST_REPLACEMENT_GLOBAL __global__
#else
#define THRUST_REPLACEMENT_HOST_DEVICE
#define THRUST_REPLACEMENT_GLOBAL
#endif
#endif

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
    copy.stream_ = reinterpret_cast<cudaStream_t>(stream);
    return copy;
  }

  Allocator *allocator_ = nullptr;
  cudaStream_t stream_ = nullptr;
};

template <>
struct execution_policy<void> {
  template <typename Stream>
  constexpr execution_policy on(Stream stream) const {
    auto copy = *this;
    copy.stream_ = reinterpret_cast<cudaStream_t>(stream);
    return copy;
  }

  using allocator_type = void;
  cudaStream_t stream_ = nullptr;
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

namespace detail {

inline void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    std::abort();
  }
}

template <typename Policy>
cudaStream_t stream(const Policy &policy) {
  return policy.stream_;
}

template <typename T>
T *raw_pointer(T *ptr) {
  return ptr;
}

template <typename T>
T *raw_pointer(::thrust::device_ptr<T> ptr) {
  return ::thrust::raw_pointer_cast(ptr);
}

template <typename Iterator>
struct iterator_value {
  using type =
      typename std::remove_cv<typename std::remove_reference<
          decltype(*std::declval<Iterator>())>::type>::type;
};

template <typename T>
struct iterator_value<::thrust::device_ptr<T>> {
  using type = T;
};

template <typename T>
struct iterator_value<T *> {
  using type = T;
};

struct less {
  template <typename T>
  THRUST_REPLACEMENT_HOST_DEVICE bool operator()(const T &lhs,
                                                 const T &rhs) const {
    return lhs < rhs;
  }
};

template <typename Policy>
typename std::enable_if<std::is_void<typename Policy::allocator_type>::value,
                        void *>::type
allocate_storage(const Policy &policy, std::size_t bytes) {
  void *ptr = nullptr;
  if (bytes != 0) {
    check_cuda(cudaMallocAsync(&ptr, bytes, stream(policy)));
  }
  return ptr;
}

template <typename Policy>
typename std::enable_if<!std::is_void<typename Policy::allocator_type>::value,
                        void *>::type
allocate_storage(const Policy &policy, std::size_t bytes) {
  if (bytes == 0) {
    return nullptr;
  }
  return policy.allocator_->allocate(static_cast<std::ptrdiff_t>(bytes));
}

template <typename Policy>
typename std::enable_if<std::is_void<typename Policy::allocator_type>::value>::type
deallocate_storage(const Policy &policy, void *ptr, std::size_t) {
  if (ptr != nullptr) {
    check_cuda(cudaFreeAsync(ptr, stream(policy)));
  }
}

template <typename Policy>
typename std::enable_if<!std::is_void<typename Policy::allocator_type>::value>::type
deallocate_storage(const Policy &policy, void *ptr, std::size_t bytes) {
  if (ptr != nullptr) {
    // Custom allocators in this codebase are not stream-ordered. Wait only on
    // the caller-provided stream before returning storage to them.
    check_cuda(cudaStreamSynchronize(stream(policy)));
    policy.allocator_->deallocate(static_cast<char *>(ptr), bytes);
  }
}

template <typename Policy>
class scoped_storage {
public:
  scoped_storage(const Policy &policy, std::size_t bytes)
      : policy_(policy), ptr_(allocate_storage(policy, bytes)), bytes_(bytes) {}

  scoped_storage(const scoped_storage &) = delete;
  scoped_storage &operator=(const scoped_storage &) = delete;

  ~scoped_storage() { deallocate_storage(policy_, ptr_, bytes_); }

  void *get() const { return ptr_; }

private:
  const Policy &policy_;
  void *ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

#ifdef __CUDACC__
template <typename InputT1, typename InputT2, typename OutputT,
          typename BinaryOperation>
THRUST_REPLACEMENT_GLOBAL void transform_binary_kernel(
    const InputT1 *first, const InputT2 *first2, OutputT *result,
    std::size_t count, BinaryOperation op) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    result[i] = op(first[i], first2[i]);
  }
}

template <typename InputT, typename OutputT, typename UnaryOperation>
THRUST_REPLACEMENT_GLOBAL void transform_unary_kernel(const InputT *first,
                                                      OutputT *result,
                                                      std::size_t count,
                                                      UnaryOperation op) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    result[i] = op(first[i]);
  }
}
#endif

template <typename Policy, typename T>
T *allocate_items(const Policy &policy, std::size_t count) {
  return static_cast<T *>(allocate_storage(policy, count * sizeof(T)));
}

template <typename Policy, typename T>
void deallocate_items(const Policy &policy, T *ptr, std::size_t count) {
  deallocate_storage(policy, ptr, count * sizeof(T));
}

} // namespace detail

template <typename Policy, typename RandomIt, typename Compare>
void sort(const Policy &policy, RandomIt first, RandomIt last, Compare comp);

template <typename Policy, typename KeyIt, typename ValueIt, typename Compare>
void sort_by_key(const Policy &policy, KeyIt keys_first, KeyIt keys_last,
                 ValueIt values_first, Compare comp);

template <typename Policy, typename KeyIt, typename ValueIt, typename Compare>
void stable_sort_by_key(const Policy &policy, KeyIt keys_first, KeyIt keys_last,
                        ValueIt values_first, Compare comp);

template <typename Policy, typename InputIt1, typename InputIt2,
          typename OutputIt, typename BinaryOperation>
OutputIt transform(const Policy &policy, InputIt1 first, InputIt1 last,
                   InputIt2 first2, OutputIt result, BinaryOperation op) {
#ifdef __CUDACC__
  std::size_t count = static_cast<std::size_t>(last - first);
  if (count != 0) {
    constexpr int block_size = 256;
    int blocks = static_cast<int>((count + block_size - 1) / block_size);
    detail::transform_binary_kernel<<<blocks, block_size, 0,
                                      detail::stream(policy)>>>(
        detail::raw_pointer(first), detail::raw_pointer(first2),
        detail::raw_pointer(result), count, op);
  }
  return result + count;
#else
  (void)policy;
  (void)first2;
  (void)op;
  return result + (last - first);
#endif
}

template <typename Policy, typename InputIt, typename OutputIt,
          typename UnaryOperation>
OutputIt transform(const Policy &policy, InputIt first, InputIt last,
                   OutputIt result, UnaryOperation op) {
#ifdef __CUDACC__
  std::size_t count = static_cast<std::size_t>(last - first);
  if (count != 0) {
    constexpr int block_size = 256;
    int blocks = static_cast<int>((count + block_size - 1) / block_size);
    detail::transform_unary_kernel<<<blocks, block_size, 0,
                                     detail::stream(policy)>>>(
        detail::raw_pointer(first), detail::raw_pointer(result), count, op);
  }
  return result + count;
#else
  (void)policy;
  (void)op;
  return result + (last - first);
#endif
}

template <typename Policy, typename InputIt, typename T,
          typename BinaryOperation>
T reduce(const Policy &policy, InputIt first, InputIt last, T init,
         BinaryOperation op) {
#ifdef __CUDACC__
  int count = static_cast<int>(last - first);
  if (count <= 0) {
    return init;
  }

  std::size_t temp_bytes = 0;
  T *device_result = detail::allocate_items<Policy, T>(policy, 1);
  detail::check_cuda(cub::DeviceReduce::Reduce(
      nullptr, temp_bytes, detail::raw_pointer(first), device_result, count, op,
      init, detail::stream(policy)));
  detail::scoped_storage<Policy> temp(policy, temp_bytes);
  detail::check_cuda(cub::DeviceReduce::Reduce(
      temp.get(), temp_bytes, detail::raw_pointer(first), device_result, count,
      op, init, detail::stream(policy)));

  T host_result{};
  detail::check_cuda(cudaMemcpyAsync(&host_result, device_result, sizeof(T),
                                     cudaMemcpyDeviceToHost,
                                     detail::stream(policy)));
  detail::deallocate_items(policy, device_result, 1);
  detail::check_cuda(cudaStreamSynchronize(detail::stream(policy)));
  return host_result;
#else
  (void)policy;
  (void)first;
  (void)last;
  (void)op;
  return init;
#endif
}

template <typename Policy, typename RandomIt>
void sort(const Policy &policy, RandomIt first, RandomIt last) {
  sort(policy, first, last, detail::less{});
}

template <typename Policy, typename RandomIt, typename Compare>
void sort(const Policy &policy, RandomIt first, RandomIt last, Compare comp) {
#ifdef __CUDACC__
  int count = static_cast<int>(last - first);
  if (count <= 1) {
    return;
  }

  std::size_t temp_bytes = 0;
  detail::check_cuda(cub::DeviceMergeSort::SortKeys(
      nullptr, temp_bytes, detail::raw_pointer(first), count, comp,
      detail::stream(policy)));
  detail::scoped_storage<Policy> temp(policy, temp_bytes);
  detail::check_cuda(cub::DeviceMergeSort::SortKeys(
      temp.get(), temp_bytes, detail::raw_pointer(first), count, comp,
      detail::stream(policy)));
#else
  (void)policy;
  (void)first;
  (void)last;
  (void)comp;
#endif
}

template <typename Policy, typename ForwardIt>
ForwardIt unique(const Policy &policy, ForwardIt first, ForwardIt last) {
#ifdef __CUDACC__
  using T = typename detail::iterator_value<ForwardIt>::type;
  int count = static_cast<int>(last - first);
  if (count <= 1) {
    return last;
  }

  T *output = detail::allocate_items<Policy, T>(policy, count);
  int *device_count = detail::allocate_items<Policy, int>(policy, 1);

  std::size_t temp_bytes = 0;
  detail::check_cuda(cub::DeviceSelect::Unique(
      nullptr, temp_bytes, detail::raw_pointer(first), output, device_count,
      count, detail::stream(policy)));
  detail::scoped_storage<Policy> temp(policy, temp_bytes);
  detail::check_cuda(cub::DeviceSelect::Unique(
      temp.get(), temp_bytes, detail::raw_pointer(first), output, device_count,
      count, detail::stream(policy)));

  int host_count = 0;
  detail::check_cuda(cudaMemcpyAsync(&host_count, device_count, sizeof(int),
                                     cudaMemcpyDeviceToHost,
                                     detail::stream(policy)));
  detail::check_cuda(cudaMemcpyAsync(detail::raw_pointer(first), output,
                                     static_cast<std::size_t>(count) * sizeof(T),
                                     cudaMemcpyDeviceToDevice,
                                     detail::stream(policy)));
  detail::deallocate_items(policy, output, count);
  detail::deallocate_items(policy, device_count, 1);
  detail::check_cuda(cudaStreamSynchronize(detail::stream(policy)));
  return first + host_count;
#else
  (void)policy;
  (void)first;
  return last;
#endif
}

template <typename Policy, typename KeyIt, typename ValueIt>
void sort_by_key(const Policy &policy, KeyIt keys_first, KeyIt keys_last,
                 ValueIt values_first) {
  sort_by_key(policy, keys_first, keys_last, values_first, detail::less{});
}

template <typename Policy, typename KeyIt, typename ValueIt, typename Compare>
void sort_by_key(const Policy &policy, KeyIt keys_first, KeyIt keys_last,
                 ValueIt values_first, Compare comp) {
#ifdef __CUDACC__
  int count = static_cast<int>(keys_last - keys_first);
  if (count <= 1) {
    return;
  }

  std::size_t temp_bytes = 0;
  detail::check_cuda(cub::DeviceMergeSort::SortPairs(
      nullptr, temp_bytes, detail::raw_pointer(keys_first),
      detail::raw_pointer(values_first), count, comp, detail::stream(policy)));
  detail::scoped_storage<Policy> temp(policy, temp_bytes);
  detail::check_cuda(cub::DeviceMergeSort::SortPairs(
      temp.get(), temp_bytes, detail::raw_pointer(keys_first),
      detail::raw_pointer(values_first), count, comp, detail::stream(policy)));
#else
  (void)policy;
  (void)keys_first;
  (void)keys_last;
  (void)values_first;
  (void)comp;
#endif
}

template <typename Policy, typename KeyIt, typename ValueIt>
void stable_sort_by_key(const Policy &policy, KeyIt keys_first, KeyIt keys_last,
                        ValueIt values_first) {
  stable_sort_by_key(policy, keys_first, keys_last, values_first,
                     detail::less{});
}

template <typename Policy, typename KeyIt, typename ValueIt, typename Compare>
void stable_sort_by_key(const Policy &policy, KeyIt keys_first, KeyIt keys_last,
                        ValueIt values_first, Compare comp) {
#ifdef __CUDACC__
  int count = static_cast<int>(keys_last - keys_first);
  if (count <= 1) {
    return;
  }

  std::size_t temp_bytes = 0;
  detail::check_cuda(cub::DeviceMergeSort::StableSortPairs(
      nullptr, temp_bytes, detail::raw_pointer(keys_first),
      detail::raw_pointer(values_first), count, comp, detail::stream(policy)));
  detail::scoped_storage<Policy> temp(policy, temp_bytes);
  detail::check_cuda(cub::DeviceMergeSort::StableSortPairs(
      temp.get(), temp_bytes, detail::raw_pointer(keys_first),
      detail::raw_pointer(values_first), count, comp, detail::stream(policy)));
#else
  (void)policy;
  (void)keys_first;
  (void)keys_last;
  (void)values_first;
  (void)comp;
#endif
}

} // namespace thrust_replacement
