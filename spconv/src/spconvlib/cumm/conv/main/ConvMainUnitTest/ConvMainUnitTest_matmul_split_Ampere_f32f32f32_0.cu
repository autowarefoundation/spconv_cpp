#include <spconvlib/cumm/conv/main/ConvMainUnitTest.h>
#include <spconvlib/cumm/common/TensorViewKernel.h>
#include <spconvlib/cumm/conv/main/cpAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK/ConvParams.h>
#include <spconvlib/cumm/conv/main/Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK/ConvKernel.h>
#include <spconvlib/cumm/conv/main/cpAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK/ConvParams.h>
#include <spconvlib/cumm/conv/main/Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK/ConvKernel.h>
#include <spconvlib/cumm/conv/main/cpAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK/ConvParams.h>
#include <spconvlib/cumm/conv/main/Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK/ConvKernel.h>
#include <spconvlib/cumm/conv/main/cpAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK/ConvParams.h>
#include <spconvlib/cumm/conv/main/Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK/ConvKernel.h>
#include <spconvlib/cumm/conv/main/cpAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK/ConvParams.h>
#include <spconvlib/cumm/conv/main/Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK/ConvKernel.h>
#include <spconvlib/cumm/conv/main/cpAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK/ConvParams.h>
#include <spconvlib/cumm/conv/main/Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK/ConvKernel.h>
#include <spconvlib/cumm/conv/main/cpAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK/ConvParams.h>
#include <spconvlib/cumm/conv/main/Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK/ConvKernel.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using ConvNVRTCParams = spconvlib::cumm::conv::kernel::ConvNVRTCParams;
using CummNVRTCLib = spconvlib::cumm::common::CummNVRTCLib;
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
using ConvParamsAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK = spconvlib::cumm::conv::main::cpAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK::ConvParams;
using ConvAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK = spconvlib::cumm::conv::main::Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK::ConvKernel;
using ConvParamsAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK = spconvlib::cumm::conv::main::cpAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK::ConvParams;
using ConvAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK = spconvlib::cumm::conv::main::Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK::ConvKernel;
using ConvParamsAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK = spconvlib::cumm::conv::main::cpAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK::ConvParams;
using ConvAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK = spconvlib::cumm::conv::main::Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK::ConvKernel;
using ConvParamsAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK = spconvlib::cumm::conv::main::cpAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK::ConvParams;
using ConvAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK = spconvlib::cumm::conv::main::Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK::ConvKernel;
using ConvParamsAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK = spconvlib::cumm::conv::main::cpAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::ConvParams;
using ConvAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK = spconvlib::cumm::conv::main::Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::ConvKernel;
using ConvParamsAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK = spconvlib::cumm::conv::main::cpAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK::ConvParams;
using ConvAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK = spconvlib::cumm::conv::main::Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK::ConvKernel;
using ConvParamsAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK = spconvlib::cumm::conv::main::cpAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK::ConvParams;
using ConvAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK = spconvlib::cumm::conv::main::Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK::ConvKernel;
void ConvMainUnitTest::matmul_split_Ampere_f32f32f32_0(tv::gemm::ConvParams params)   {
  
  // auto rtxtimer = tv::CPUTimer<>();
  // auto ev1 = tv::CUDAEvent("wtf").record();
  static_assert(3 == CUMM_MAXIMUM_NVRTC_CONV_NDIM, "error");
  int groups = 1;
  bool found = false;
  auto& algo_desp = params.conv_algo_desp;
  auto dacc = tv::DType(algo_desp.dacc);
  auto dcomp = tv::DType(algo_desp.dcomp);
  tv::gemm::ConvOpType op_type = static_cast<tv::gemm::ConvOpType>(algo_desp.op_type);
  int split_k_slices = params.split_k_slices;
  auto& workspace = params.workspace;
  auto& input = params.input;
  auto& weight = params.weight;
  auto& output = params.output;
  auto& output_add = params.output_add;
  auto& bias = params.bias;
  auto& scale = params.scale;
  auto& indices = params.indices;
  auto& mask = params.mask;
  auto& mask_argsort = params.mask_argsort;
  auto& mask_output = params.mask_output;
  auto& padding = params.padding;
  auto& stride = params.stride;
  auto& dilation = params.dilation;
  auto& mask_width = params.mask_width;
  auto& evtimer = params.timer;
  int io_dim = algo_desp.mask_sparse ? 2 : algo_desp.ndim + 2;
  int weight_ndim = algo_desp.mask_sparse ? 3 : algo_desp.ndim + 2;
  int dim_start =  algo_desp.layout_w == tv::gemm::ConvLayoutType::kChannelFirst ? 2 : 1;
  int ndim = algo_desp.ndim;
  TV_ASSERT_RT_ERR(input.ndim() == io_dim, "error");
  TV_ASSERT_RT_ERR(weight.ndim() == weight_ndim, "error");
  TV_ASSERT_RT_ERR(output.ndim() == io_dim, "error");
  if (!(algo_desp.split_k_serial() || algo_desp.split_k_parallel()) && split_k_slices > 1){
      TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
  }
  int kernel_volume = 1;
  int N = input.dim(0);
  int K = weight.dim(0);
  int C = algo_desp.layout_i == tv::gemm::ConvLayoutType::kChannelFirst ? input.dim(1) : input.dim(io_dim - 1);
  int K2 = algo_desp.layout_o == tv::gemm::ConvLayoutType::kChannelFirst ? output.dim(1) : output.dim(io_dim - 1);
  TV_ASSERT_RT_ERR(K2 == K, "error");
  tv::array<int, 3> mnk;
  auto inv_indices = tv::gemm::gemm_abc_012_to_iwo(tv::gemm::ConvOpType(algo_desp.op_type));
  std::array<tv::Tensor, 3> conv_inputs{input, weight, output};
  auto& a_ten = conv_inputs[inv_indices[0]];
  auto& b_ten = conv_inputs[inv_indices[1]];
  auto& c_ten = conv_inputs[inv_indices[2]];
  auto& nvrtc_params = params.nvrtc_params;
  tv::gemm::ConvNVRTCParams kernel_params;
  tv::gemm::SparseConvNVRTCParams sp_kernel_params;
  kernel_params.ptr_A = a_ten.const_raw_data();
  kernel_params.ptr_B = b_ten.const_raw_data();
  kernel_params.ptr_C = c_ten.raw_data();
  if (!algo_desp.is_int8_inference){
      TV_ASSERT_INVALID_ARG(output_add.empty(), "only int8 inference support output_add not empty ")
  }else{
      TV_ASSERT_RT_ERR(!bias.empty() && !scale.empty(), "int8 inference must have both scale and bias");
  }
  if (output_add.empty()){
      kernel_params.ptr_D = algo_desp.is_int8_inference ? c_ten.const_raw_data() : (bias.empty() ? c_ten.const_raw_data() : bias.const_raw_data());
  }else{
      TV_ASSERT_RT_ERR(output_add.dtype() == output.dtype() && output_add.shape() == output.shape(),
          "output and output_add must have same dtype and shape", output_add.dtype(), output.dtype(),
          output_add.shape(), output.shape());
      kernel_params.ptr_D = output_add.const_raw_data();
  }
  kernel_params.bias_pointer = bias.empty() ? nullptr : bias.const_raw_data();
  kernel_params.scale_pointer = scale.empty() ? nullptr : scale.const_raw_data();
  kernel_params.alpha = params.alpha;
  kernel_params.beta = params.beta;
  kernel_params.ndim = ndim;
  kernel_params.d_is_bias = !bias.empty();
  kernel_params.act_alpha = params.act_alpha;
  kernel_params.act_beta = params.act_beta;
  kernel_params.act_type = static_cast<int>(params.act_type);
  sp_kernel_params.ptr_A = kernel_params.ptr_A;
  sp_kernel_params.ptr_B = kernel_params.ptr_B;
  sp_kernel_params.ptr_C = kernel_params.ptr_C;
  sp_kernel_params.ptr_D = kernel_params.ptr_D;
  sp_kernel_params.scale_pointer = kernel_params.scale_pointer;
  sp_kernel_params.bias_pointer = kernel_params.bias_pointer;
  sp_kernel_params.alpha = kernel_params.alpha;
  sp_kernel_params.beta = kernel_params.beta;
  sp_kernel_params.ndim = kernel_params.ndim;
  sp_kernel_params.d_is_bias = !bias.empty();
  sp_kernel_params.act_alpha = kernel_params.act_alpha;
  sp_kernel_params.act_beta = kernel_params.act_beta;
  sp_kernel_params.act_type = kernel_params.act_type;
  constexpr int int_max = std::numeric_limits<int32_t>::max();
  if (algo_desp.mask_sparse){
      if (algo_desp.op_type == tv::gemm::ConvOpType::kBackwardWeight){
          TV_ASSERT_RT_ERR(mask_width > 0 && mask_width % algo_desp.tile_shape[2] == 0, "error");
      }
      TV_ASSERT_RT_ERR(!indices.empty(), "error");
      TV_ASSERT_RT_ERR(!mask.empty(), "error");
      TV_ASSERT_RT_ERR(!mask_argsort.empty(), "error");
      kernel_volume = weight.dim(dim_start);
      tv::check_shape(indices, {kernel_volume, -1});
      N = indices.dim(1);
      if (algo_desp.op_type == tv::gemm::ConvOpType::kBackwardWeight){
          TV_ASSERT_RT_ERR(N == output.dim(0), "error");
          TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * tv::bit_size(algo_desp.dtype_b) / 8 < int_max, 
              "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
          TV_ASSERT_RT_ERR(int64_t(N) * int64_t(K) * tv::bit_size(algo_desp.dtype_a) / 8 < int_max, 
              "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
      }else if (algo_desp.op_type == tv::gemm::ConvOpType::kForward){
          TV_ASSERT_RT_ERR(N == output.dim(0), "error");
          TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * tv::bit_size(algo_desp.dtype_a) / 8 < int_max, 
              "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
      }else{
              TV_ASSERT_RT_ERR(int64_t(N) * int64_t(K) * tv::bit_size(algo_desp.dtype_a) / 8 < int_max, 
                  "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
              TV_ASSERT_RT_ERR(N == input.dim(0), "error");
      }
      mnk = tv::gemm::implicit_gemm_mnk(tv::gemm::ConvOpType(algo_desp.op_type), N, C, K, kernel_volume, -1, -1, true);
  }else{
      TV_ASSERT_RT_ERR(algo_desp.ndim <= 3, "ndim too large for nvrtc");
      tv::array<int, 3> ksize, padding_arr, stride_arr, dilation_arr, input_dims, output_dims;
      TV_ASSERT_RT_ERR(ndim == padding.size() && ndim == stride.size() && ndim == dilation.size(), "error");
      for (int i = dim_start; i < dim_start + ndim; ++i){
          ksize[i - dim_start] = weight.dim(i);
          input_dims[i - dim_start] = input.dim(i);
          output_dims[i - dim_start] = output.dim(i);
      }
      for (int i = 0; i < ndim; ++i){
          padding_arr[i] = padding[i];
          stride_arr[i] = stride[i];
          dilation_arr[i] = dilation[i];
      }
      kernel_volume = 1;
      int in_prod = 1;
      int out_prod = 1;
      for (int i = 0; i < ndim; ++i){
          kernel_volume *= ksize[i];
          in_prod *= input_dims[i];
          out_prod *= output_dims[i];
      }
      mnk = tv::gemm::implicit_gemm_mnk(tv::gemm::ConvOpType(algo_desp.op_type), N, C, K, kernel_volume, in_prod, out_prod, false);
      kernel_params.input_dims = input_dims;
      kernel_params.output_dims = output_dims;
      kernel_params.ksize = ksize;
      kernel_params.padding = padding_arr;
      kernel_params.stride = stride_arr;
      kernel_params.dilation = dilation_arr;
  }
  TV_ASSERT_RT_ERR(algo_desp.supported(mnk[0], mnk[1], mnk[2], C, K, mask_width), "error");
  int workspace_size = algo_desp.query_conv_workspace_size(mnk[0], mnk[1], mnk[2], split_k_slices, kernel_volume);
  auto ctx = tv::Context();
  ctx.set_cuda_stream(reinterpret_cast<cudaStream_t>(params.stream));
  if (workspace_size > 0){
      if (!workspace.empty()){
          workspace.zero_(ctx);
          TV_ASSERT_RT_ERR(workspace.nbytes() >= workspace_size, 
              "workspace at least", workspace_size, "bytes.");
      }else{
          workspace = tv::empty({workspace_size}, tv::uint8, 0);
          workspace.zero_(ctx);
      }
  }
  void* workspace_ptr = nullptr;
  if (!workspace.empty()){
      workspace_ptr = workspace.raw_data();
  }
  if (nvrtc_params.cumodule){
      TV_ASSERT_RT_ERR(!nvrtc_params.kernel_name.empty(), "you must provide name of your kernel");
      kernel_params.N = N;
      kernel_params.C = C;
      kernel_params.K = K;
      kernel_params.kernel_volume = kernel_volume;
      kernel_params.mode = static_cast<int>(tv::gemm::ConvMode::kCrossCorrelation);
      kernel_params.split_k_slices = split_k_slices;
      kernel_params.groups = groups;
      kernel_params.workspace = workspace_ptr;
      sp_kernel_params.N = kernel_params.N;
      sp_kernel_params.C = kernel_params.C;
      sp_kernel_params.K = kernel_params.K;
      sp_kernel_params.kernel_volume = kernel_params.kernel_volume;
      sp_kernel_params.mode = kernel_params.mode;
      sp_kernel_params.split_k_slices = kernel_params.split_k_slices;
      sp_kernel_params.groups = kernel_params.groups;
      sp_kernel_params.workspace = kernel_params.workspace;
      tv::array<int, 3> grid_dims_arr;
      if (algo_desp.mask_sparse){
          sp_kernel_params.mask_out_ptr = mask_output.empty() ? nullptr : mask_output.data_ptr<uint32_t>();
          sp_kernel_params.mask_width = mask_width;
          sp_kernel_params.mask_ptr = mask.data_ptr<const uint32_t>();
          sp_kernel_params.reverse_mask = params.reverse_mask;
          sp_kernel_params.mask_filter = params.mask_filter;
          sp_kernel_params.indice_ptr = indices.data_ptr<const int32_t>();
          sp_kernel_params.mask_argsort_ptr = mask_argsort.data_ptr<const int32_t>();
          grid_dims_arr = tv::gemm::get_spconv_logical_tile_count(mnk[0], mnk[1], mnk[2], 
                          algo_desp.tile_shape[0], algo_desp.tile_shape[1], split_k_slices, kernel_volume, algo_desp.op_type);
      }else{
          grid_dims_arr = tv::gemm::get_logical_tile_count(mnk[0], mnk[1], mnk[2], 
                          algo_desp.tile_shape[0], algo_desp.tile_shape[1], split_k_slices);
      }
      dim3 grid_dims;
      grid_dims.x = grid_dims_arr[0];
      grid_dims.y = grid_dims_arr[1];
      grid_dims.z = grid_dims_arr[2];
      if (algo_desp.op_type == tv::gemm::ConvOpType::kBackwardWeight && algo_desp.mask_sparse){
          int num_reduced_mask = tv::div_up(sp_kernel_params.N, sp_kernel_params.mask_width);
          TV_ASSERT_RT_ERR(mask.dim(0) >= num_reduced_mask, "error");
      }
      std::string algo_name;
      if (evtimer.enable()){
          algo_name = algo_desp.__repr__();
      }
      auto kernel = nvrtc_params.cumodule->kernel(nvrtc_params.kernel_name);
      auto& driver = nvrtc_params.cumodule->get_driver_wrapper();
      cudaError_t result;
      if (nvrtc_params.smem_size > 0){
          if (nvrtc_params.smem_size >= (48 << 10)) {
              TV_CUDA_RESULT_CHECK(driver.cuDrvFuncSetAttribute(kernel,
                                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                              nvrtc_params.smem_size));
              TV_CUDA_RESULT_CHECK(driver.cuDrvFuncSetAttribute(
                  kernel,
                  CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100));
          }
      }
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(params.stream);
      void* kernel_params_ptr;
      if (algo_desp.mask_sparse){
          kernel_params_ptr = &sp_kernel_params;
      }else{
          kernel_params_ptr = &kernel_params;
      }
      // auto ev2 = tv::CUDAEvent("wtf").record();
      // ev1.sync();
      // ev2.sync();
      // tv::ssprint("prep time", tv::CUDAEvent::duration(ev1, ev2));
      if (nvrtc_params.mode == 2){
          tv::CUDAKernelTimerGuard timerguard(algo_name, evtimer, stream);
              std::vector<void*> args{kernel_params_ptr, &grid_dims, &params.stream};
          TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(kernel, 1, 1, 1, 
              1, 1, 1, 0, stream, args.data(), 0));
      }else if (nvrtc_params.mode == 3){
          // use kernel-cpu-kernel
          auto init_kernel = nvrtc_params.cumodule->kernel(nvrtc_params.init_kernel_name);
          tv::Tensor temp_data = nvrtc_params.param_storage;
          if (nvrtc_params.param_storage.empty()){
              temp_data = tv::empty({nvrtc_params.param_size}, tv::uint8, 0);
          }else{
              TV_ASSERT_RT_ERR(temp_data.nbytes() >= nvrtc_params.param_size, "your params storage too small");
          }
          void* raw_data_ptr;
          void* temp_data_ptr = temp_data.raw_data();
          tv::Tensor temp_data_cpu = nvrtc_params.param_storage_cpu;
          {
              tv::CUDAKernelTimerGuard timerguard(algo_name + "/init", evtimer, stream);
              std::vector<void*> args{kernel_params_ptr, &temp_data_ptr};
              TV_CUDA_RESULT_CHECK(driver.cuDrvLaunchKernel(init_kernel, 1, 1, 1, 32, 1, 1, 0, stream, args.data(), 0));
              if (nvrtc_params.param_storage_cpu.empty()){
                  temp_data_cpu = temp_data.cpu(ctx);
              }else{
                  temp_data_cpu.copy_(temp_data, ctx);
              }
              // we must sync here because following kernel launch requires cpu data.
              checkCudaErrors(cudaStreamSynchronize(stream));
              raw_data_ptr = temp_data_cpu.raw_data();
          }
          {
              tv::CUDAKernelTimerGuard timerguard(algo_name, evtimer, stream);
              std::vector<void*> args{raw_data_ptr};
              TV_CUDA_RESULT_CHECK(driver.cuDrvLaunchKernel(kernel, grid_dims.x, grid_dims.y, grid_dims.z, 
                  nvrtc_params.num_threads, 1, 1, nvrtc_params.smem_size, stream, args.data(), 0));
          }
      }else if (nvrtc_params.mode == 1){
          tv::CUDAKernelTimerGuard timerguard(algo_name, evtimer, stream);
          std::vector<void*> args{kernel_params_ptr};
          TV_CUDA_RESULT_CHECK(driver.cuDrvLaunchKernel(kernel, grid_dims.x, grid_dims.y, grid_dims.z, 
              nvrtc_params.num_threads, 1, 1, nvrtc_params.smem_size, stream, args.data(), 0));
      }else if (nvrtc_params.mode == 4){
          auto init_kernel = nvrtc_params.cumodule->kernel(nvrtc_params.init_kernel_name);
          tv::Tensor temp_data = nvrtc_params.param_storage;
          if (nvrtc_params.param_storage.empty()){
              temp_data = tv::empty({nvrtc_params.param_size}, tv::uint8, 0);
          }else{
              TV_ASSERT_RT_ERR(temp_data.nbytes() >= nvrtc_params.param_size, "your params storage too small");
          }
          void* temp_data_ptr = temp_data.raw_data();
          {
              tv::CUDAKernelTimerGuard timerguard(algo_name + "/init", evtimer, stream);
              std::vector<void*> args{kernel_params_ptr, &temp_data_ptr};
              TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(init_kernel, 1, 1, 1, 32, 1, 1, 0, stream, args.data(), 0));
          }
          {
              tv::CUDAKernelTimerGuard timerguard(algo_name, evtimer, stream);
              auto ptr = nvrtc_params.cumodule->get_global_ptr(nvrtc_params.constant_name);
              auto constant_ten = tv::from_blob(ptr, {nvrtc_params.param_size}, tv::uint8, 0);
              constant_ten.copy_(temp_data, ctx);
              std::vector<void*> args{};
              TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(kernel, grid_dims.x, grid_dims.y, grid_dims.z, 
                  nvrtc_params.num_threads, 1, 1, nvrtc_params.smem_size, stream, nullptr, 0));
          }
      }else{
          TV_THROW_RT_ERR("not implemented");
      }
      TV_CHECK_CUDA_ERR_V2(algo_desp.__repr__(), "error with params", input.shape(), output.shape(), weight.shape());
      return;
  }
  if (algo_desp.trans_a() == false && algo_desp.trans_b() == true && algo_desp.trans_c() == false){
    if (algo_desp.tile_shape == std::array<int, 3>{32, 32, 16}){
      if (algo_desp.warp_tile_shape == std::array<int, 3>{16, 16, 16}){
        if (algo_desp.num_stage == 2 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 0){
              if (algo_desp.tensorop == std::array<int, 3>{16, 8, 8}){
                if (algo_desp.ndim == 3 && static_cast<int>(algo_desp.op_type) == 0 && static_cast<int>(algo_desp.iter_algo) == 1){
                  if (static_cast<int>(algo_desp.layout_i) == 1 && static_cast<int>(algo_desp.layout_w) == 1 && static_cast<int>(algo_desp.layout_o) == 1){
                    if (algo_desp.interleave_i == 1 && algo_desp.interleave_w == 1 && algo_desp.interleave_o == 1){
                      if (algo_desp.mask_sparse == true && algo_desp.increment_k_first == true && algo_desp.is_int8_inference == false && algo_desp.dynamic_mask == false){
                        TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
                        // Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK
                        found = true;
                        bool d_is_bias = !bias.empty();
                        TV_ASSERT_RT_ERR(!indices.empty(), "error");
                        TV_ASSERT_RT_ERR(!mask.empty(), "error");
                        TV_ASSERT_RT_ERR(!mask_argsort.empty(), "error");
                        int kernel_volume = weight.dim(1);
                        tv::check_shape(indices, {kernel_volume, -1});
                        N = indices.dim(1);
                        cpAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK::ConvProblem problem(N, C, K, kernel_volume, 
                            tv::gemm::ConvMode::kCrossCorrelation, split_k_slices, groups);
                        TV_ASSERT_RT_ERR(N == output.dim(0), "error");
                        TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * 32 / 8 < std::numeric_limits<int32_t>::max(), 
                            "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                        if (!algo_desp.is_int8_inference){
                            TV_ASSERT_INVALID_ARG(output_add.empty(), "only int8 inference support output_add not empty ")
                        }
                        auto source_ptr = algo_desp.is_int8_inference ? c_ten.data_ptr<const float>() : (bias.empty() ? c_ten.data_ptr<const float>() : bias.data_ptr<const float>());
                        if (!output_add.empty()){
                            source_ptr = output_add.data_ptr<const float>();
                        }
                        ConvParamsAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK ker_params(problem, a_ten.data_ptr<const float>(), b_ten.data_ptr<const float>(), c_ten.data_ptr<float>(), source_ptr, mask.data_ptr<const uint32_t>(), mask_argsort.data_ptr<const int32_t>(), indices.data_ptr<const int32_t>(), mask_output.empty() ? nullptr : mask_output.data_ptr<uint32_t>(), params.mask_filter, params.reverse_mask, float(params.alpha), float(params.beta), float(params.act_alpha), float(params.act_beta), params.act_type, 1, d_is_bias);
                        tv::cuda::Launch launcher(ker_params.grid_dims, dim3(128),
                                                    8192, reinterpret_cast<cudaStream_t>(params.stream));
                        cudaError_t result;
                        if (8192 >= (48 << 10)) {
                            result = cudaFuncSetAttribute(Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK::conv_kernel,
                                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                            8192);
                            TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                            result = cudaFuncSetAttribute(
                                Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK::conv_kernel,
                                cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                            TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                        }
                        auto timer = tv::CUDATimer(params.verbose);
                        // tv::ssprint("CPU Time", rtxtimer.report() / 1000.0);
                        {
                            tv::CUDAKernelTimerGuard timerguard("Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                            launcher(Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK::conv_kernel, ker_params);
                        }
                        TV_CHECK_CUDA_ERR_V2("Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK", "error with params", input.shape(), weight.shape(), output.shape(), 
                            indices.shape(), mask.shape(), mask_argsort.shape(), mask_output.shape(), mask_width);
                        if (params.verbose){
                            cudaFuncAttributes attr;
                            checkCudaErrors(
                                cudaFuncGetAttributes(&attr, Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK::conv_kernel));
                            tv::ssprint("Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A0T1688_200_C301LLL_SK kernel num regs:", attr.numRegs, "time:", timer.report() / 1000.0);
                        }
                        return;
                      }
                    }
                  }
                }
              }
            }
            if (algo_desp.access_per_vector == 1){
              if (algo_desp.tensorop == std::array<int, 3>{16, 8, 8}){
                if (algo_desp.ndim == 3 && static_cast<int>(algo_desp.op_type) == 0 && static_cast<int>(algo_desp.iter_algo) == 1){
                  if (static_cast<int>(algo_desp.layout_i) == 1 && static_cast<int>(algo_desp.layout_w) == 1 && static_cast<int>(algo_desp.layout_o) == 1){
                    if (algo_desp.interleave_i == 1 && algo_desp.interleave_w == 1 && algo_desp.interleave_o == 1){
                      if (algo_desp.mask_sparse == true && algo_desp.increment_k_first == true && algo_desp.is_int8_inference == false && algo_desp.dynamic_mask == false){
                        TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
                        // Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK
                        found = true;
                        bool d_is_bias = !bias.empty();
                        TV_ASSERT_RT_ERR(!indices.empty(), "error");
                        TV_ASSERT_RT_ERR(!mask.empty(), "error");
                        TV_ASSERT_RT_ERR(!mask_argsort.empty(), "error");
                        int kernel_volume = weight.dim(1);
                        tv::check_shape(indices, {kernel_volume, -1});
                        N = indices.dim(1);
                        cpAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK::ConvProblem problem(N, C, K, kernel_volume, 
                            tv::gemm::ConvMode::kCrossCorrelation, split_k_slices, groups);
                        TV_ASSERT_RT_ERR(N == output.dim(0), "error");
                        TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * 32 / 8 < std::numeric_limits<int32_t>::max(), 
                            "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                        if (!algo_desp.is_int8_inference){
                            TV_ASSERT_INVALID_ARG(output_add.empty(), "only int8 inference support output_add not empty ")
                        }
                        auto source_ptr = algo_desp.is_int8_inference ? c_ten.data_ptr<const float>() : (bias.empty() ? c_ten.data_ptr<const float>() : bias.data_ptr<const float>());
                        if (!output_add.empty()){
                            source_ptr = output_add.data_ptr<const float>();
                        }
                        ConvParamsAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK ker_params(problem, a_ten.data_ptr<const float>(), b_ten.data_ptr<const float>(), c_ten.data_ptr<float>(), source_ptr, mask.data_ptr<const uint32_t>(), mask_argsort.data_ptr<const int32_t>(), indices.data_ptr<const int32_t>(), mask_output.empty() ? nullptr : mask_output.data_ptr<uint32_t>(), params.mask_filter, params.reverse_mask, float(params.alpha), float(params.beta), float(params.act_alpha), float(params.act_beta), params.act_type, 1, d_is_bias);
                        tv::cuda::Launch launcher(ker_params.grid_dims, dim3(128),
                                                    8192, reinterpret_cast<cudaStream_t>(params.stream));
                        cudaError_t result;
                        if (8192 >= (48 << 10)) {
                            result = cudaFuncSetAttribute(Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK::conv_kernel,
                                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                            8192);
                            TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                            result = cudaFuncSetAttribute(
                                Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK::conv_kernel,
                                cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                            TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                        }
                        auto timer = tv::CUDATimer(params.verbose);
                        // tv::ssprint("CPU Time", rtxtimer.report() / 1000.0);
                        {
                            tv::CUDAKernelTimerGuard timerguard("Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                            launcher(Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK::conv_kernel, ker_params);
                        }
                        TV_CHECK_CUDA_ERR_V2("Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK", "error with params", input.shape(), weight.shape(), output.shape(), 
                            indices.shape(), mask.shape(), mask_argsort.shape(), mask_output.shape(), mask_width);
                        if (params.verbose){
                            cudaFuncAttributes attr;
                            checkCudaErrors(
                                cudaFuncGetAttributes(&attr, Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK::conv_kernel));
                            tv::ssprint("Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_200_C301LLL_SK kernel num regs:", attr.numRegs, "time:", timer.report() / 1000.0);
                        }
                        return;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        if (algo_desp.num_stage == 3 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              if (algo_desp.tensorop == std::array<int, 3>{16, 8, 8}){
                if (algo_desp.ndim == 3 && static_cast<int>(algo_desp.op_type) == 0 && static_cast<int>(algo_desp.iter_algo) == 1){
                  if (static_cast<int>(algo_desp.layout_i) == 1 && static_cast<int>(algo_desp.layout_w) == 1 && static_cast<int>(algo_desp.layout_o) == 1){
                    if (algo_desp.interleave_i == 1 && algo_desp.interleave_w == 1 && algo_desp.interleave_o == 1){
                      if (algo_desp.mask_sparse == true && algo_desp.increment_k_first == true && algo_desp.is_int8_inference == false && algo_desp.dynamic_mask == false){
                        TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
                        // Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK
                        found = true;
                        bool d_is_bias = !bias.empty();
                        TV_ASSERT_RT_ERR(!indices.empty(), "error");
                        TV_ASSERT_RT_ERR(!mask.empty(), "error");
                        TV_ASSERT_RT_ERR(!mask_argsort.empty(), "error");
                        int kernel_volume = weight.dim(1);
                        tv::check_shape(indices, {kernel_volume, -1});
                        N = indices.dim(1);
                        cpAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK::ConvProblem problem(N, C, K, kernel_volume, 
                            tv::gemm::ConvMode::kCrossCorrelation, split_k_slices, groups);
                        TV_ASSERT_RT_ERR(N == output.dim(0), "error");
                        TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * 32 / 8 < std::numeric_limits<int32_t>::max(), 
                            "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                        if (!algo_desp.is_int8_inference){
                            TV_ASSERT_INVALID_ARG(output_add.empty(), "only int8 inference support output_add not empty ")
                        }
                        auto source_ptr = algo_desp.is_int8_inference ? c_ten.data_ptr<const float>() : (bias.empty() ? c_ten.data_ptr<const float>() : bias.data_ptr<const float>());
                        if (!output_add.empty()){
                            source_ptr = output_add.data_ptr<const float>();
                        }
                        ConvParamsAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK ker_params(problem, a_ten.data_ptr<const float>(), b_ten.data_ptr<const float>(), c_ten.data_ptr<float>(), source_ptr, mask.data_ptr<const uint32_t>(), mask_argsort.data_ptr<const int32_t>(), indices.data_ptr<const int32_t>(), mask_output.empty() ? nullptr : mask_output.data_ptr<uint32_t>(), params.mask_filter, params.reverse_mask, float(params.alpha), float(params.beta), float(params.act_alpha), float(params.act_beta), params.act_type, 1, d_is_bias);
                        tv::cuda::Launch launcher(ker_params.grid_dims, dim3(128),
                                                    12288, reinterpret_cast<cudaStream_t>(params.stream));
                        cudaError_t result;
                        if (12288 >= (48 << 10)) {
                            result = cudaFuncSetAttribute(Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK::conv_kernel,
                                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                            12288);
                            TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                            result = cudaFuncSetAttribute(
                                Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK::conv_kernel,
                                cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                            TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                        }
                        auto timer = tv::CUDATimer(params.verbose);
                        // tv::ssprint("CPU Time", rtxtimer.report() / 1000.0);
                        {
                            tv::CUDAKernelTimerGuard timerguard("Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                            launcher(Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK::conv_kernel, ker_params);
                        }
                        TV_CHECK_CUDA_ERR_V2("Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK", "error with params", input.shape(), weight.shape(), output.shape(), 
                            indices.shape(), mask.shape(), mask_argsort.shape(), mask_output.shape(), mask_width);
                        if (params.verbose){
                            cudaFuncAttributes attr;
                            checkCudaErrors(
                                cudaFuncGetAttributes(&attr, Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK::conv_kernel));
                            tv::ssprint("Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_300_C301LLL_SK kernel num regs:", attr.numRegs, "time:", timer.report() / 1000.0);
                        }
                        return;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        if (algo_desp.num_stage == 4 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              if (algo_desp.tensorop == std::array<int, 3>{16, 8, 8}){
                if (algo_desp.ndim == 3 && static_cast<int>(algo_desp.op_type) == 0 && static_cast<int>(algo_desp.iter_algo) == 1){
                  if (static_cast<int>(algo_desp.layout_i) == 1 && static_cast<int>(algo_desp.layout_w) == 1 && static_cast<int>(algo_desp.layout_o) == 1){
                    if (algo_desp.interleave_i == 1 && algo_desp.interleave_w == 1 && algo_desp.interleave_o == 1){
                      if (algo_desp.mask_sparse == true && algo_desp.increment_k_first == true && algo_desp.is_int8_inference == false && algo_desp.dynamic_mask == false){
                        TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
                        // Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK
                        found = true;
                        bool d_is_bias = !bias.empty();
                        TV_ASSERT_RT_ERR(!indices.empty(), "error");
                        TV_ASSERT_RT_ERR(!mask.empty(), "error");
                        TV_ASSERT_RT_ERR(!mask_argsort.empty(), "error");
                        int kernel_volume = weight.dim(1);
                        tv::check_shape(indices, {kernel_volume, -1});
                        N = indices.dim(1);
                        cpAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK::ConvProblem problem(N, C, K, kernel_volume, 
                            tv::gemm::ConvMode::kCrossCorrelation, split_k_slices, groups);
                        TV_ASSERT_RT_ERR(N == output.dim(0), "error");
                        TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * 32 / 8 < std::numeric_limits<int32_t>::max(), 
                            "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                        if (!algo_desp.is_int8_inference){
                            TV_ASSERT_INVALID_ARG(output_add.empty(), "only int8 inference support output_add not empty ")
                        }
                        auto source_ptr = algo_desp.is_int8_inference ? c_ten.data_ptr<const float>() : (bias.empty() ? c_ten.data_ptr<const float>() : bias.data_ptr<const float>());
                        if (!output_add.empty()){
                            source_ptr = output_add.data_ptr<const float>();
                        }
                        ConvParamsAmpere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK ker_params(problem, a_ten.data_ptr<const float>(), b_ten.data_ptr<const float>(), c_ten.data_ptr<float>(), source_ptr, mask.data_ptr<const uint32_t>(), mask_argsort.data_ptr<const int32_t>(), indices.data_ptr<const int32_t>(), mask_output.empty() ? nullptr : mask_output.data_ptr<uint32_t>(), params.mask_filter, params.reverse_mask, float(params.alpha), float(params.beta), float(params.act_alpha), float(params.act_beta), params.act_type, 1, d_is_bias);
                        tv::cuda::Launch launcher(ker_params.grid_dims, dim3(128),
                                                    16384, reinterpret_cast<cudaStream_t>(params.stream));
                        cudaError_t result;
                        if (16384 >= (48 << 10)) {
                            result = cudaFuncSetAttribute(Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK::conv_kernel,
                                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                            16384);
                            TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                            result = cudaFuncSetAttribute(
                                Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK::conv_kernel,
                                cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                            TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                        }
                        auto timer = tv::CUDATimer(params.verbose);
                        // tv::ssprint("CPU Time", rtxtimer.report() / 1000.0);
                        {
                            tv::CUDAKernelTimerGuard timerguard("Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                            launcher(Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK::conv_kernel, ker_params);
                        }
                        TV_CHECK_CUDA_ERR_V2("Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK", "error with params", input.shape(), weight.shape(), output.shape(), 
                            indices.shape(), mask.shape(), mask_argsort.shape(), mask_output.shape(), mask_width);
                        if (params.verbose){
                            cudaFuncAttributes attr;
                            checkCudaErrors(
                                cudaFuncGetAttributes(&attr, Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK::conv_kernel));
                            tv::ssprint("Ampere_f32f32f32f32f32tnt_m32n32k16m16n16k16A1T1688_400_C301LLL_SK kernel num regs:", attr.numRegs, "time:", timer.report() / 1000.0);
                        }
                        return;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    if (algo_desp.tile_shape == std::array<int, 3>{64, 64, 32}){
      if (algo_desp.warp_tile_shape == std::array<int, 3>{32, 32, 32}){
        if (algo_desp.num_stage == 2 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              if (algo_desp.tensorop == std::array<int, 3>{16, 8, 8}){
                if (algo_desp.ndim == 3 && static_cast<int>(algo_desp.op_type) == 0 && static_cast<int>(algo_desp.iter_algo) == 1){
                  if (static_cast<int>(algo_desp.layout_i) == 1 && static_cast<int>(algo_desp.layout_w) == 1 && static_cast<int>(algo_desp.layout_o) == 1){
                    if (algo_desp.interleave_i == 1 && algo_desp.interleave_w == 1 && algo_desp.interleave_o == 1){
                      if (algo_desp.mask_sparse == true && algo_desp.increment_k_first == true && algo_desp.is_int8_inference == false && algo_desp.dynamic_mask == false){
                        TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
                        // Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK
                        found = true;
                        bool d_is_bias = !bias.empty();
                        TV_ASSERT_RT_ERR(!indices.empty(), "error");
                        TV_ASSERT_RT_ERR(!mask.empty(), "error");
                        TV_ASSERT_RT_ERR(!mask_argsort.empty(), "error");
                        int kernel_volume = weight.dim(1);
                        tv::check_shape(indices, {kernel_volume, -1});
                        N = indices.dim(1);
                        cpAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::ConvProblem problem(N, C, K, kernel_volume, 
                            tv::gemm::ConvMode::kCrossCorrelation, split_k_slices, groups);
                        TV_ASSERT_RT_ERR(N == output.dim(0), "error");
                        TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * 32 / 8 < std::numeric_limits<int32_t>::max(), 
                            "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                        if (!algo_desp.is_int8_inference){
                            TV_ASSERT_INVALID_ARG(output_add.empty(), "only int8 inference support output_add not empty ")
                        }
                        auto source_ptr = algo_desp.is_int8_inference ? c_ten.data_ptr<const float>() : (bias.empty() ? c_ten.data_ptr<const float>() : bias.data_ptr<const float>());
                        if (!output_add.empty()){
                            source_ptr = output_add.data_ptr<const float>();
                        }
                        ConvParamsAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK ker_params(problem, a_ten.data_ptr<const float>(), b_ten.data_ptr<const float>(), c_ten.data_ptr<float>(), source_ptr, mask.data_ptr<const uint32_t>(), mask_argsort.data_ptr<const int32_t>(), indices.data_ptr<const int32_t>(), mask_output.empty() ? nullptr : mask_output.data_ptr<uint32_t>(), params.mask_filter, params.reverse_mask, float(params.alpha), float(params.beta), float(params.act_alpha), float(params.act_beta), params.act_type, 1, d_is_bias);
                        tv::cuda::Launch launcher(ker_params.grid_dims, dim3(128),
                                                    32768, reinterpret_cast<cudaStream_t>(params.stream));
                        cudaError_t result;
                        if (32768 >= (48 << 10)) {
                            result = cudaFuncSetAttribute(Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::conv_kernel,
                                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                            32768);
                            TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                            result = cudaFuncSetAttribute(
                                Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::conv_kernel,
                                cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                            TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                        }
                        auto timer = tv::CUDATimer(params.verbose);
                        // tv::ssprint("CPU Time", rtxtimer.report() / 1000.0);
                        {
                            tv::CUDAKernelTimerGuard timerguard("Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                            launcher(Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::conv_kernel, ker_params);
                        }
                        TV_CHECK_CUDA_ERR_V2("Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK", "error with params", input.shape(), weight.shape(), output.shape(), 
                            indices.shape(), mask.shape(), mask_argsort.shape(), mask_output.shape(), mask_width);
                        if (params.verbose){
                            cudaFuncAttributes attr;
                            checkCudaErrors(
                                cudaFuncGetAttributes(&attr, Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK::conv_kernel));
                            tv::ssprint("Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_200_C301LLL_SK kernel num regs:", attr.numRegs, "time:", timer.report() / 1000.0);
                        }
                        return;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        if (algo_desp.num_stage == 3 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              if (algo_desp.tensorop == std::array<int, 3>{16, 8, 8}){
                if (algo_desp.ndim == 3 && static_cast<int>(algo_desp.op_type) == 0 && static_cast<int>(algo_desp.iter_algo) == 1){
                  if (static_cast<int>(algo_desp.layout_i) == 1 && static_cast<int>(algo_desp.layout_w) == 1 && static_cast<int>(algo_desp.layout_o) == 1){
                    if (algo_desp.interleave_i == 1 && algo_desp.interleave_w == 1 && algo_desp.interleave_o == 1){
                      if (algo_desp.mask_sparse == true && algo_desp.increment_k_first == true && algo_desp.is_int8_inference == false && algo_desp.dynamic_mask == false){
                        TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
                        // Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK
                        found = true;
                        bool d_is_bias = !bias.empty();
                        TV_ASSERT_RT_ERR(!indices.empty(), "error");
                        TV_ASSERT_RT_ERR(!mask.empty(), "error");
                        TV_ASSERT_RT_ERR(!mask_argsort.empty(), "error");
                        int kernel_volume = weight.dim(1);
                        tv::check_shape(indices, {kernel_volume, -1});
                        N = indices.dim(1);
                        cpAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK::ConvProblem problem(N, C, K, kernel_volume, 
                            tv::gemm::ConvMode::kCrossCorrelation, split_k_slices, groups);
                        TV_ASSERT_RT_ERR(N == output.dim(0), "error");
                        TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * 32 / 8 < std::numeric_limits<int32_t>::max(), 
                            "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                        if (!algo_desp.is_int8_inference){
                            TV_ASSERT_INVALID_ARG(output_add.empty(), "only int8 inference support output_add not empty ")
                        }
                        auto source_ptr = algo_desp.is_int8_inference ? c_ten.data_ptr<const float>() : (bias.empty() ? c_ten.data_ptr<const float>() : bias.data_ptr<const float>());
                        if (!output_add.empty()){
                            source_ptr = output_add.data_ptr<const float>();
                        }
                        ConvParamsAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK ker_params(problem, a_ten.data_ptr<const float>(), b_ten.data_ptr<const float>(), c_ten.data_ptr<float>(), source_ptr, mask.data_ptr<const uint32_t>(), mask_argsort.data_ptr<const int32_t>(), indices.data_ptr<const int32_t>(), mask_output.empty() ? nullptr : mask_output.data_ptr<uint32_t>(), params.mask_filter, params.reverse_mask, float(params.alpha), float(params.beta), float(params.act_alpha), float(params.act_beta), params.act_type, 1, d_is_bias);
                        tv::cuda::Launch launcher(ker_params.grid_dims, dim3(128),
                                                    49152, reinterpret_cast<cudaStream_t>(params.stream));
                        cudaError_t result;
                        if (49152 >= (48 << 10)) {
                            result = cudaFuncSetAttribute(Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK::conv_kernel,
                                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                            49152);
                            TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                            result = cudaFuncSetAttribute(
                                Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK::conv_kernel,
                                cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                            TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                        }
                        auto timer = tv::CUDATimer(params.verbose);
                        // tv::ssprint("CPU Time", rtxtimer.report() / 1000.0);
                        {
                            tv::CUDAKernelTimerGuard timerguard("Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                            launcher(Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK::conv_kernel, ker_params);
                        }
                        TV_CHECK_CUDA_ERR_V2("Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK", "error with params", input.shape(), weight.shape(), output.shape(), 
                            indices.shape(), mask.shape(), mask_argsort.shape(), mask_output.shape(), mask_width);
                        if (params.verbose){
                            cudaFuncAttributes attr;
                            checkCudaErrors(
                                cudaFuncGetAttributes(&attr, Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK::conv_kernel));
                            tv::ssprint("Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_300_C301LLL_SK kernel num regs:", attr.numRegs, "time:", timer.report() / 1000.0);
                        }
                        return;
                      }
                    }
                  }
                }
              }
            }
          }
        }
        if (algo_desp.num_stage == 4 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              if (algo_desp.tensorop == std::array<int, 3>{16, 8, 8}){
                if (algo_desp.ndim == 3 && static_cast<int>(algo_desp.op_type) == 0 && static_cast<int>(algo_desp.iter_algo) == 1){
                  if (static_cast<int>(algo_desp.layout_i) == 1 && static_cast<int>(algo_desp.layout_w) == 1 && static_cast<int>(algo_desp.layout_o) == 1){
                    if (algo_desp.interleave_i == 1 && algo_desp.interleave_w == 1 && algo_desp.interleave_o == 1){
                      if (algo_desp.mask_sparse == true && algo_desp.increment_k_first == true && algo_desp.is_int8_inference == false && algo_desp.dynamic_mask == false){
                        TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
                        // Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK
                        found = true;
                        bool d_is_bias = !bias.empty();
                        TV_ASSERT_RT_ERR(!indices.empty(), "error");
                        TV_ASSERT_RT_ERR(!mask.empty(), "error");
                        TV_ASSERT_RT_ERR(!mask_argsort.empty(), "error");
                        int kernel_volume = weight.dim(1);
                        tv::check_shape(indices, {kernel_volume, -1});
                        N = indices.dim(1);
                        cpAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK::ConvProblem problem(N, C, K, kernel_volume, 
                            tv::gemm::ConvMode::kCrossCorrelation, split_k_slices, groups);
                        TV_ASSERT_RT_ERR(N == output.dim(0), "error");
                        TV_ASSERT_RT_ERR(int64_t(N) * int64_t(C) * 32 / 8 < std::numeric_limits<int32_t>::max(), 
                            "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
                        if (!algo_desp.is_int8_inference){
                            TV_ASSERT_INVALID_ARG(output_add.empty(), "only int8 inference support output_add not empty ")
                        }
                        auto source_ptr = algo_desp.is_int8_inference ? c_ten.data_ptr<const float>() : (bias.empty() ? c_ten.data_ptr<const float>() : bias.data_ptr<const float>());
                        if (!output_add.empty()){
                            source_ptr = output_add.data_ptr<const float>();
                        }
                        ConvParamsAmpere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK ker_params(problem, a_ten.data_ptr<const float>(), b_ten.data_ptr<const float>(), c_ten.data_ptr<float>(), source_ptr, mask.data_ptr<const uint32_t>(), mask_argsort.data_ptr<const int32_t>(), indices.data_ptr<const int32_t>(), mask_output.empty() ? nullptr : mask_output.data_ptr<uint32_t>(), params.mask_filter, params.reverse_mask, float(params.alpha), float(params.beta), float(params.act_alpha), float(params.act_beta), params.act_type, 1, d_is_bias);
                        tv::cuda::Launch launcher(ker_params.grid_dims, dim3(128),
                                                    65536, reinterpret_cast<cudaStream_t>(params.stream));
                        cudaError_t result;
                        if (65536 >= (48 << 10)) {
                            result = cudaFuncSetAttribute(Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK::conv_kernel,
                                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                            65536);
                            TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                            result = cudaFuncSetAttribute(
                                Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK::conv_kernel,
                                cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                            TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                        }
                        auto timer = tv::CUDATimer(params.verbose);
                        // tv::ssprint("CPU Time", rtxtimer.report() / 1000.0);
                        {
                            tv::CUDAKernelTimerGuard timerguard("Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                            launcher(Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK::conv_kernel, ker_params);
                        }
                        TV_CHECK_CUDA_ERR_V2("Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK", "error with params", input.shape(), weight.shape(), output.shape(), 
                            indices.shape(), mask.shape(), mask_argsort.shape(), mask_output.shape(), mask_width);
                        if (params.verbose){
                            cudaFuncAttributes attr;
                            checkCudaErrors(
                                cudaFuncGetAttributes(&attr, Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK::conv_kernel));
                            tv::ssprint("Ampere_f32f32f32f32f32tnt_m64n64k32m32n32k32A1T1688_400_C301LLL_SK kernel num regs:", attr.numRegs, "time:", timer.report() / 1000.0);
                        }
                        return;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  if (!found){
      TV_THROW_INVALID_ARG("Can't Found Algorithm for params:", algo_desp.__repr__(), tv::dtype_str(input.dtype()), 
          tv::dtype_str(weight.dtype()), tv::dtype_str(output.dtype()), tv::dtype_str(dacc), 
          tv::dtype_str(dcomp));
  }
}
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib