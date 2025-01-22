#include <spconvlib/cumm/gemm/main/GemmMainUnitTest.h>
#include <spconvlib/cumm/common/TensorViewKernel.h>
#include <spconvlib/cumm/gemm/main/gpSimt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1/GemmParams.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1/GemmKernel.h>
#include <spconvlib/cumm/gemm/main/gpSimt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1/GemmParams.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1/GemmKernel.h>
#include <spconvlib/cumm/gemm/main/gpSimt_f16f16f16f32f32ttt_m32n64k32m32n32k8A1_200_S1/GemmParams.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m32n64k32m32n32k8A1_200_S1/GemmKernel.h>
#include <spconvlib/cumm/gemm/main/gpSimt_f16f16f16f32f32tnt_m32n64k32m32n32k8A1_200_S1/GemmParams.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32tnt_m32n64k32m32n32k8A1_200_S1/GemmKernel.h>
#include <spconvlib/cumm/gemm/main/gpSimt_f16f16f16f32f32ttt_m32n32k32m32n32k8A1_200_S1/GemmParams.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m32n32k32m32n32k8A1_200_S1/GemmKernel.h>
#include <spconvlib/cumm/gemm/main/gpSimt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_S1/GemmParams.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_S1/GemmKernel.h>
#include <spconvlib/cumm/gemm/main/gpSimt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1/GemmParams.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1/GemmKernel.h>
#include <spconvlib/cumm/gemm/main/gpSimt_f16f16f16f32f32tnt_m64n128k16m32n64k8A1_200_S1/GemmParams.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32tnt_m64n128k16m32n64k8A1_200_S1/GemmKernel.h>
#include <spconvlib/cumm/gemm/main/gpSimt_f16f16f16f32f32ttt_m64n64k8m32n32k8A1_200_S1/GemmParams.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32ttt_m64n64k8m32n32k8A1_200_S1/GemmKernel.h>
#include <spconvlib/cumm/gemm/main/gpSimt_f16f16f16f32f32tnt_m64n64k8m32n32k8A1_200_S1/GemmParams.h>
#include <spconvlib/cumm/gemm/main/Simt_f16f16f16f32f32tnt_m64n64k8m32n32k8A1_200_S1/GemmKernel.h>
namespace spconvlib {
namespace cumm {
namespace gemm {
namespace main {
using TensorView = spconvlib::cumm::common::TensorView;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmBasicHost = spconvlib::cumm::common::GemmBasicHost;
using TensorViewKernel = spconvlib::cumm::common::TensorViewKernel;
using GemmParamsSimt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1 = spconvlib::cumm::gemm::main::gpSimt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::GemmParams;
using GemmSimt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1 = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::GemmKernel;
using GemmParamsSimt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1 = spconvlib::cumm::gemm::main::gpSimt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1::GemmParams;
using GemmSimt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1 = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1::GemmKernel;
using GemmParamsSimt_f16f16f16f32f32ttt_m32n64k32m32n32k8A1_200_S1 = spconvlib::cumm::gemm::main::gpSimt_f16f16f16f32f32ttt_m32n64k32m32n32k8A1_200_S1::GemmParams;
using GemmSimt_f16f16f16f32f32ttt_m32n64k32m32n32k8A1_200_S1 = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m32n64k32m32n32k8A1_200_S1::GemmKernel;
using GemmParamsSimt_f16f16f16f32f32tnt_m32n64k32m32n32k8A1_200_S1 = spconvlib::cumm::gemm::main::gpSimt_f16f16f16f32f32tnt_m32n64k32m32n32k8A1_200_S1::GemmParams;
using GemmSimt_f16f16f16f32f32tnt_m32n64k32m32n32k8A1_200_S1 = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32tnt_m32n64k32m32n32k8A1_200_S1::GemmKernel;
using GemmParamsSimt_f16f16f16f32f32ttt_m32n32k32m32n32k8A1_200_S1 = spconvlib::cumm::gemm::main::gpSimt_f16f16f16f32f32ttt_m32n32k32m32n32k8A1_200_S1::GemmParams;
using GemmSimt_f16f16f16f32f32ttt_m32n32k32m32n32k8A1_200_S1 = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m32n32k32m32n32k8A1_200_S1::GemmKernel;
using GemmParamsSimt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_S1 = spconvlib::cumm::gemm::main::gpSimt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_S1::GemmParams;
using GemmSimt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_S1 = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_S1::GemmKernel;
using GemmParamsSimt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1 = spconvlib::cumm::gemm::main::gpSimt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1::GemmParams;
using GemmSimt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1 = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1::GemmKernel;
using GemmParamsSimt_f16f16f16f32f32tnt_m64n128k16m32n64k8A1_200_S1 = spconvlib::cumm::gemm::main::gpSimt_f16f16f16f32f32tnt_m64n128k16m32n64k8A1_200_S1::GemmParams;
using GemmSimt_f16f16f16f32f32tnt_m64n128k16m32n64k8A1_200_S1 = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32tnt_m64n128k16m32n64k8A1_200_S1::GemmKernel;
using GemmParamsSimt_f16f16f16f32f32ttt_m64n64k8m32n32k8A1_200_S1 = spconvlib::cumm::gemm::main::gpSimt_f16f16f16f32f32ttt_m64n64k8m32n32k8A1_200_S1::GemmParams;
using GemmSimt_f16f16f16f32f32ttt_m64n64k8m32n32k8A1_200_S1 = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32ttt_m64n64k8m32n32k8A1_200_S1::GemmKernel;
using GemmParamsSimt_f16f16f16f32f32tnt_m64n64k8m32n32k8A1_200_S1 = spconvlib::cumm::gemm::main::gpSimt_f16f16f16f32f32tnt_m64n64k8m32n32k8A1_200_S1::GemmParams;
using GemmSimt_f16f16f16f32f32tnt_m64n64k8m32n32k8A1_200_S1 = spconvlib::cumm::gemm::main::Simt_f16f16f16f32f32tnt_m64n64k8m32n32k8A1_200_S1::GemmKernel;
void GemmMainUnitTest::matmul_split_Simt_f16f16f16_1(tv::gemm::GemmParams params)   {
  
  params.check_valid();
  auto& algo_desp = params.algo_desp;
  bool found = false;
  auto dacc = tv::DType(algo_desp.dacc);
  auto dcomp = tv::DType(algo_desp.dcomp);
  auto a = params.a;
  auto b = params.b;
  auto c = params.c;
  auto d = params.d;
  if (d.empty()){
      d = c; // TODO fix this
  }
  auto ta = algo_desp.trans_a();
  auto tb = algo_desp.trans_b();
  auto tc = algo_desp.trans_c();
  tv::check_shape(a, {-1, -1});
  tv::check_shape(b, {-1, -1});
  tv::check_shape(c, {-1, -1});
  tv::check_eq_device(a, b, c);
  tv::Tensor a_ten = a;
  tv::Tensor b_ten = b;
  tv::Tensor c_ten = c;
  tv::Tensor d_ten = d;
  auto trans_a = ta;
  auto trans_b = tb;
  auto trans_c = tc;
  if (tc) {
      trans_a = !trans_a;
      trans_b = !trans_b;
      std::swap(trans_a, trans_b);
      std::swap(a_ten, b_ten);
  }
  int split_k_slices = params.split_k_slices;
  auto workspace = params.workspace;
  auto a_inds = params.a_inds;
  auto c_inds = params.c_inds;
  auto b_inds = params.b_inds;
  auto& evtimer = params.timer;
  if (!(algo_desp.split_k_serial() || algo_desp.split_k_parallel()) && split_k_slices > 1){
      TV_ASSERT_RT_ERR("algo don't support splitk but you provide split_k_slices > 1.", split_k_slices);
  }
  int m, n, k, k2;
  constexpr int int_max = std::numeric_limits<int32_t>::max();
  if (algo_desp.shuffle_type == tv::gemm::ShuffleStrideType::kShuffleAC){
      TV_ASSERT_RT_ERR(!trans_a, "a of shuffle AB must be row major");
      if (!a_inds.empty()){
          m = a_inds.dim(0);
      }else{
          m = a.dim(0);
      }
      TV_ASSERT_RT_ERR(int64_t(a.dim(0)) * int64_t(a.dim(1)) * tv::bit_size(algo_desp.dtype_a) / 8 < int_max, 
          "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
      k = a_ten.dim(int(!trans_a));
      k2 = b_ten.dim(int(trans_b));
      n = b_ten.dim(int(!trans_b) );
      if (trans_c){
          tv::check_shape(c_ten, {-1, m});
      }else{
          tv::check_shape(c_ten, {-1, n});
      }
  }else if (algo_desp.shuffle_type == tv::gemm::ShuffleStrideType::kShuffleAB){
      TV_ASSERT_RT_ERR(trans_a && !trans_b, "shuffle AB must be nt, i.e. backward weight");
      m = a_ten.dim(int(trans_a));
      k = a_inds.dim(0);
      k2 = b_inds.dim(0);
      n = b_ten.dim(int(!trans_b) );
      TV_ASSERT_RT_ERR(int64_t(a.dim(0)) * int64_t(a.dim(1)) * tv::bit_size(algo_desp.dtype_a)/ 8 < int_max, 
          "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
      TV_ASSERT_RT_ERR(int64_t(b.dim(0)) * int64_t(b.dim(1)) * tv::bit_size(algo_desp.dtype_b) / 8 < int_max, 
          "your data exceed int32 range. this will be fixed in cumm + nvrtc (spconv 2.2/2.3).");
      if (trans_c){
          tv::check_shape(c_ten, {n, m});
      }else{
          tv::check_shape(c_ten, {m, n});
      }
  }else{
      m = a_ten.dim(int(trans_a));
      k = a_ten.dim(int(!trans_a));
      k2 = b_ten.dim(int(trans_b));
      n = b_ten.dim(int(!trans_b) );
      if (trans_c){
          tv::check_shape(c_ten, {n, m});
      }else{
          tv::check_shape(c_ten, {m, n});
      }
  }
  TV_ASSERT_INVALID_ARG(algo_desp.supported(m, n, k), "this m, n, k isn't supported due to misaligned contiguous dim.")
  TV_ASSERT_INVALID_ARG(k == k2, "error");
  if (d.ndim() == 1){
      TV_ASSERT_RT_ERR(d.dim(0) == n, "d must be a valid bias");
  }
  int workspace_size = algo_desp.query_workspace_size(m, n, k, split_k_slices);
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
  auto& nvrtc_params = params.nvrtc_params;
  if (nvrtc_params.cumodule){
      TV_ASSERT_RT_ERR(nvrtc_params.kernel_name != "", "you must provide name of your kernel");
      tv::gemm::GemmNVRTCParams kernel_params;
      if (algo_desp.shuffle_type == tv::gemm::ShuffleStrideType::kShuffleAC){
          const int* a_ptr = nullptr;
          if (!a_inds.empty()){
              a_ptr = a_inds.data_ptr<const int>();
          }
          TV_ASSERT_RT_ERR(!c_inds.empty(), "c must not empty");
          auto indice_ptr = c_inds.data_ptr<const int>();
          kernel_params = tv::gemm::GemmNVRTCParams{m, n, k, a_ten.const_raw_data(),  b_ten.const_raw_data(),  
              c_ten.raw_data(), d_ten.raw_data(), 
              a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0),
              a_ptr, indice_ptr, d.ndim() == 1 ? nullptr : indice_ptr, 
              float(params.alpha), float(params.beta), 
              float(params.act_alpha), float(params.act_beta), 
              static_cast<int>(params.act_type),
              split_k_slices, workspace_ptr};
      }else if (algo_desp.shuffle_type == tv::gemm::ShuffleStrideType::kShuffleAB){
          TV_ASSERT_RT_ERR(!a_inds.empty() && !b_inds.empty(), "error");
          kernel_params = tv::gemm::GemmNVRTCParams{m, n, k, a_ten.const_raw_data(),  b_ten.const_raw_data(),  
              c_ten.raw_data(), d_ten.raw_data(), 
              a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0),
              a_inds.data_ptr<const int>(), b_inds.data_ptr<const int>(), nullptr,
              float(params.alpha), float(params.beta), 
              float(params.act_alpha), float(params.act_beta), 
              static_cast<int>(params.act_type),
              split_k_slices, workspace_ptr};
      }else{
          kernel_params = tv::gemm::GemmNVRTCParams{m, n, k, a_ten.const_raw_data(),  b_ten.const_raw_data(),  
              c_ten.raw_data(), c_ten.raw_data(), 
              a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0),
              nullptr, nullptr, nullptr,
              float(params.alpha), float(params.beta), 
              float(params.act_alpha), float(params.act_beta), 
              static_cast<int>(params.act_type),
              split_k_slices, workspace_ptr};
      }
      std::string algo_name;
      if (evtimer.enable()){
          algo_name = algo_desp.__repr__();
      }
      auto grid_dims_arr = tv::gemm::get_logical_tile_count(m, n, k, algo_desp.tile_shape[0], algo_desp.tile_shape[1], split_k_slices);
      TV_ASSERT_RT_ERR(grid_dims_arr[0] != 0 && grid_dims_arr[1] != 0 && grid_dims_arr[2] != 0, "unexpected error",
          m, n, k, algo_desp.tile_shape[0], algo_desp.tile_shape[1], split_k_slices);
      dim3 grid_dims;
      grid_dims.x = grid_dims_arr[0];
      grid_dims.y = grid_dims_arr[1];
      grid_dims.z = grid_dims_arr[2];
      cudaStream_t stream = reinterpret_cast<cudaStream_t>(params.stream);
      auto kernel = nvrtc_params.cumodule->kernel(nvrtc_params.kernel_name);
      if (nvrtc_params.mode == 2){
          tv::CUDAKernelTimerGuard timerguard(algo_name, evtimer, stream);
          std::vector<void*> args{&kernel_params, &grid_dims, &params.stream};
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
              std::vector<void*> args{&kernel_params, &temp_data_ptr};
              TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(init_kernel, 1, 1, 1, 32, 1, 1, 0, stream, args.data(), 0));
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
              // tv::ssprint(reinterpret_cast<tv::array<int, 4>*>(raw_data_ptr)[0]);
              // tv::ssprint(grid_dims.x, grid_dims.y, grid_dims.z, temp_data.size(), temp_data_cpu.size());
              TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(kernel, grid_dims.x, grid_dims.y, grid_dims.z, 
                  nvrtc_params.num_threads, 1, 1, nvrtc_params.smem_size, stream, args.data(), 0));
          }
      }else if (nvrtc_params.mode == 1){
          tv::CUDAKernelTimerGuard timerguard(algo_name, evtimer, stream);
          std::vector<void*> args{&kernel_params};
          TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(kernel, grid_dims.x, grid_dims.y, grid_dims.z, 
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
              std::vector<void*> args{&kernel_params, &temp_data_ptr};
              TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(init_kernel, 1, 1, 1, 32, 1, 1, 0, stream, args.data(), 0));
          }
          {
              tv::CUDAKernelTimerGuard timerguard(algo_name, evtimer, stream);
              auto ptr = nvrtc_params.cumodule->get_global_ptr(nvrtc_params.constant_name);
              auto constant_ten = tv::from_blob(ptr, {nvrtc_params.param_size}, tv::uint8, 0);
              constant_ten.copy_(temp_data, ctx);
              std::vector<void*> args{};
              TV_CUDA_RESULT_CHECK(nvrtc_params.cumodule->cuDrvLaunchKernel(kernel, grid_dims.x, grid_dims.y, grid_dims.z, 
                  nvrtc_params.num_threads, 1, 1, nvrtc_params.smem_size, stream, args.data(), 0));
          }
      }else{
          TV_THROW_RT_ERR("not implemented");
      }
      TV_CHECK_CUDA_ERR_V2(algo_desp.__repr__(), "error with params", a.shape(), b.shape(), c.shape());
      return;
  }
  if (algo_desp.trans_a() == false && algo_desp.trans_b() == false && algo_desp.trans_c() == false){
    if (algo_desp.tile_shape == std::array<int, 3>{128, 128, 8}){
      if (algo_desp.warp_tile_shape == std::array<int, 3>{32, 64, 8}){
        if (algo_desp.num_stage == 2 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              found = true;
              const int* a_ptr = nullptr;
              if (!a_inds.empty()){
                  a_ptr = a_inds.data_ptr<const int>();
              }
              TV_ASSERT_RT_ERR(!c_inds.empty(), "c must not empty");
              // tv::ssprint(d.ndim() == 1 ? 0 : d_ten.stride(0), (d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>()) == nullptr, "WTF");
              GemmParamsSimt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1 kernel_params(
                  m, n, k, a_ten.data_ptr<const tv::half_t>(), b_ten.data_ptr<const tv::half_t>(),
                  c_ten.data_ptr<tv::half_t>(), d_ten.data_ptr<tv::half_t>(), 
                  a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0), 
                  a_ptr, c_inds.data_ptr<const int>(), d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>(), 
                  float(params.alpha), float(params.beta), 
                  float(params.act_alpha), float(params.act_beta),
                  params.act_type,
                  split_k_slices);
              tv::cuda::Launch launcher(kernel_params.grid_dims, dim3(256),
                                          10304, reinterpret_cast<cudaStream_t>(params.stream));
              cudaError_t result;
              if (10304 >= (48 << 10)) {
                  result = cudaFuncSetAttribute(Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::gemm_kernel,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  10304);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                  result = cudaFuncSetAttribute(
                      Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::gemm_kernel,
                      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
              }
              {
                  tv::CUDAKernelTimerGuard timerguard("Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                  launcher(Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1::gemm_kernel, kernel_params);
              }
              TV_CHECK_CUDA_ERR_V2("Simt_f16f16f16f32f32ttt_m128n128k8m32n64k8A1_200_S1", "error with params", a.shape(), b.shape(), c.shape());
              return;
            }
          }
        }
      }
    }
    if (algo_desp.tile_shape == std::array<int, 3>{32, 64, 32}){
      if (algo_desp.warp_tile_shape == std::array<int, 3>{32, 32, 8}){
        if (algo_desp.num_stage == 2 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              found = true;
              const int* a_ptr = nullptr;
              if (!a_inds.empty()){
                  a_ptr = a_inds.data_ptr<const int>();
              }
              TV_ASSERT_RT_ERR(!c_inds.empty(), "c must not empty");
              // tv::ssprint(d.ndim() == 1 ? 0 : d_ten.stride(0), (d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>()) == nullptr, "WTF");
              GemmParamsSimt_f16f16f16f32f32ttt_m32n64k32m32n32k8A1_200_S1 kernel_params(
                  m, n, k, a_ten.data_ptr<const tv::half_t>(), b_ten.data_ptr<const tv::half_t>(),
                  c_ten.data_ptr<tv::half_t>(), d_ten.data_ptr<tv::half_t>(), 
                  a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0), 
                  a_ptr, c_inds.data_ptr<const int>(), d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>(), 
                  float(params.alpha), float(params.beta), 
                  float(params.act_alpha), float(params.act_beta),
                  params.act_type,
                  split_k_slices);
              tv::cuda::Launch launcher(kernel_params.grid_dims, dim3(256),
                                          12544, reinterpret_cast<cudaStream_t>(params.stream));
              cudaError_t result;
              if (12544 >= (48 << 10)) {
                  result = cudaFuncSetAttribute(Simt_f16f16f16f32f32ttt_m32n64k32m32n32k8A1_200_S1::gemm_kernel,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  12544);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                  result = cudaFuncSetAttribute(
                      Simt_f16f16f16f32f32ttt_m32n64k32m32n32k8A1_200_S1::gemm_kernel,
                      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
              }
              {
                  tv::CUDAKernelTimerGuard timerguard("Simt_f16f16f16f32f32ttt_m32n64k32m32n32k8A1_200_S1", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                  launcher(Simt_f16f16f16f32f32ttt_m32n64k32m32n32k8A1_200_S1::gemm_kernel, kernel_params);
              }
              TV_CHECK_CUDA_ERR_V2("Simt_f16f16f16f32f32ttt_m32n64k32m32n32k8A1_200_S1", "error with params", a.shape(), b.shape(), c.shape());
              return;
            }
          }
        }
      }
    }
    if (algo_desp.tile_shape == std::array<int, 3>{32, 32, 32}){
      if (algo_desp.warp_tile_shape == std::array<int, 3>{32, 32, 8}){
        if (algo_desp.num_stage == 2 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              found = true;
              const int* a_ptr = nullptr;
              if (!a_inds.empty()){
                  a_ptr = a_inds.data_ptr<const int>();
              }
              TV_ASSERT_RT_ERR(!c_inds.empty(), "c must not empty");
              // tv::ssprint(d.ndim() == 1 ? 0 : d_ten.stride(0), (d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>()) == nullptr, "WTF");
              GemmParamsSimt_f16f16f16f32f32ttt_m32n32k32m32n32k8A1_200_S1 kernel_params(
                  m, n, k, a_ten.data_ptr<const tv::half_t>(), b_ten.data_ptr<const tv::half_t>(),
                  c_ten.data_ptr<tv::half_t>(), d_ten.data_ptr<tv::half_t>(), 
                  a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0), 
                  a_ptr, c_inds.data_ptr<const int>(), d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>(), 
                  float(params.alpha), float(params.beta), 
                  float(params.act_alpha), float(params.act_beta),
                  params.act_type,
                  split_k_slices);
              tv::cuda::Launch launcher(kernel_params.grid_dims, dim3(128),
                                          8448, reinterpret_cast<cudaStream_t>(params.stream));
              cudaError_t result;
              if (8448 >= (48 << 10)) {
                  result = cudaFuncSetAttribute(Simt_f16f16f16f32f32ttt_m32n32k32m32n32k8A1_200_S1::gemm_kernel,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  8448);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                  result = cudaFuncSetAttribute(
                      Simt_f16f16f16f32f32ttt_m32n32k32m32n32k8A1_200_S1::gemm_kernel,
                      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
              }
              {
                  tv::CUDAKernelTimerGuard timerguard("Simt_f16f16f16f32f32ttt_m32n32k32m32n32k8A1_200_S1", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                  launcher(Simt_f16f16f16f32f32ttt_m32n32k32m32n32k8A1_200_S1::gemm_kernel, kernel_params);
              }
              TV_CHECK_CUDA_ERR_V2("Simt_f16f16f16f32f32ttt_m32n32k32m32n32k8A1_200_S1", "error with params", a.shape(), b.shape(), c.shape());
              return;
            }
          }
        }
      }
    }
    if (algo_desp.tile_shape == std::array<int, 3>{64, 128, 16}){
      if (algo_desp.warp_tile_shape == std::array<int, 3>{32, 64, 8}){
        if (algo_desp.num_stage == 2 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              found = true;
              const int* a_ptr = nullptr;
              if (!a_inds.empty()){
                  a_ptr = a_inds.data_ptr<const int>();
              }
              TV_ASSERT_RT_ERR(!c_inds.empty(), "c must not empty");
              // tv::ssprint(d.ndim() == 1 ? 0 : d_ten.stride(0), (d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>()) == nullptr, "WTF");
              GemmParamsSimt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1 kernel_params(
                  m, n, k, a_ten.data_ptr<const tv::half_t>(), b_ten.data_ptr<const tv::half_t>(),
                  c_ten.data_ptr<tv::half_t>(), d_ten.data_ptr<tv::half_t>(), 
                  a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0), 
                  a_ptr, c_inds.data_ptr<const int>(), d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>(), 
                  float(params.alpha), float(params.beta), 
                  float(params.act_alpha), float(params.act_beta),
                  params.act_type,
                  split_k_slices);
              tv::cuda::Launch launcher(kernel_params.grid_dims, dim3(256),
                                          12544, reinterpret_cast<cudaStream_t>(params.stream));
              cudaError_t result;
              if (12544 >= (48 << 10)) {
                  result = cudaFuncSetAttribute(Simt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1::gemm_kernel,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  12544);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                  result = cudaFuncSetAttribute(
                      Simt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1::gemm_kernel,
                      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
              }
              {
                  tv::CUDAKernelTimerGuard timerguard("Simt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                  launcher(Simt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1::gemm_kernel, kernel_params);
              }
              TV_CHECK_CUDA_ERR_V2("Simt_f16f16f16f32f32ttt_m64n128k16m32n64k8A1_200_S1", "error with params", a.shape(), b.shape(), c.shape());
              return;
            }
          }
        }
      }
    }
    if (algo_desp.tile_shape == std::array<int, 3>{64, 64, 8}){
      if (algo_desp.warp_tile_shape == std::array<int, 3>{32, 32, 8}){
        if (algo_desp.num_stage == 2 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              found = true;
              const int* a_ptr = nullptr;
              if (!a_inds.empty()){
                  a_ptr = a_inds.data_ptr<const int>();
              }
              TV_ASSERT_RT_ERR(!c_inds.empty(), "c must not empty");
              // tv::ssprint(d.ndim() == 1 ? 0 : d_ten.stride(0), (d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>()) == nullptr, "WTF");
              GemmParamsSimt_f16f16f16f32f32ttt_m64n64k8m32n32k8A1_200_S1 kernel_params(
                  m, n, k, a_ten.data_ptr<const tv::half_t>(), b_ten.data_ptr<const tv::half_t>(),
                  c_ten.data_ptr<tv::half_t>(), d_ten.data_ptr<tv::half_t>(), 
                  a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0), 
                  a_ptr, c_inds.data_ptr<const int>(), d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>(), 
                  float(params.alpha), float(params.beta), 
                  float(params.act_alpha), float(params.act_beta),
                  params.act_type,
                  split_k_slices);
              tv::cuda::Launch launcher(kernel_params.grid_dims, dim3(128),
                                          4352, reinterpret_cast<cudaStream_t>(params.stream));
              cudaError_t result;
              if (4352 >= (48 << 10)) {
                  result = cudaFuncSetAttribute(Simt_f16f16f16f32f32ttt_m64n64k8m32n32k8A1_200_S1::gemm_kernel,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  4352);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                  result = cudaFuncSetAttribute(
                      Simt_f16f16f16f32f32ttt_m64n64k8m32n32k8A1_200_S1::gemm_kernel,
                      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
              }
              {
                  tv::CUDAKernelTimerGuard timerguard("Simt_f16f16f16f32f32ttt_m64n64k8m32n32k8A1_200_S1", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                  launcher(Simt_f16f16f16f32f32ttt_m64n64k8m32n32k8A1_200_S1::gemm_kernel, kernel_params);
              }
              TV_CHECK_CUDA_ERR_V2("Simt_f16f16f16f32f32ttt_m64n64k8m32n32k8A1_200_S1", "error with params", a.shape(), b.shape(), c.shape());
              return;
            }
          }
        }
      }
    }
  }
  if (algo_desp.trans_a() == false && algo_desp.trans_b() == true && algo_desp.trans_c() == false){
    if (algo_desp.tile_shape == std::array<int, 3>{128, 128, 8}){
      if (algo_desp.warp_tile_shape == std::array<int, 3>{32, 64, 8}){
        if (algo_desp.num_stage == 2 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              found = true;
              const int* a_ptr = nullptr;
              if (!a_inds.empty()){
                  a_ptr = a_inds.data_ptr<const int>();
              }
              TV_ASSERT_RT_ERR(!c_inds.empty(), "c must not empty");
              // tv::ssprint(d.ndim() == 1 ? 0 : d_ten.stride(0), (d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>()) == nullptr, "WTF");
              GemmParamsSimt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1 kernel_params(
                  m, n, k, a_ten.data_ptr<const tv::half_t>(), b_ten.data_ptr<const tv::half_t>(),
                  c_ten.data_ptr<tv::half_t>(), d_ten.data_ptr<tv::half_t>(), 
                  a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0), 
                  a_ptr, c_inds.data_ptr<const int>(), d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>(), 
                  float(params.alpha), float(params.beta), 
                  float(params.act_alpha), float(params.act_beta),
                  params.act_type,
                  split_k_slices);
              tv::cuda::Launch launcher(kernel_params.grid_dims, dim3(256),
                                          10304, reinterpret_cast<cudaStream_t>(params.stream));
              cudaError_t result;
              if (10304 >= (48 << 10)) {
                  result = cudaFuncSetAttribute(Simt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1::gemm_kernel,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  10304);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                  result = cudaFuncSetAttribute(
                      Simt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1::gemm_kernel,
                      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
              }
              {
                  tv::CUDAKernelTimerGuard timerguard("Simt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                  launcher(Simt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1::gemm_kernel, kernel_params);
              }
              TV_CHECK_CUDA_ERR_V2("Simt_f16f16f16f32f32tnt_m128n128k8m32n64k8A1_200_S1", "error with params", a.shape(), b.shape(), c.shape());
              return;
            }
          }
        }
      }
    }
    if (algo_desp.tile_shape == std::array<int, 3>{32, 64, 32}){
      if (algo_desp.warp_tile_shape == std::array<int, 3>{32, 32, 8}){
        if (algo_desp.num_stage == 2 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              found = true;
              const int* a_ptr = nullptr;
              if (!a_inds.empty()){
                  a_ptr = a_inds.data_ptr<const int>();
              }
              TV_ASSERT_RT_ERR(!c_inds.empty(), "c must not empty");
              // tv::ssprint(d.ndim() == 1 ? 0 : d_ten.stride(0), (d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>()) == nullptr, "WTF");
              GemmParamsSimt_f16f16f16f32f32tnt_m32n64k32m32n32k8A1_200_S1 kernel_params(
                  m, n, k, a_ten.data_ptr<const tv::half_t>(), b_ten.data_ptr<const tv::half_t>(),
                  c_ten.data_ptr<tv::half_t>(), d_ten.data_ptr<tv::half_t>(), 
                  a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0), 
                  a_ptr, c_inds.data_ptr<const int>(), d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>(), 
                  float(params.alpha), float(params.beta), 
                  float(params.act_alpha), float(params.act_beta),
                  params.act_type,
                  split_k_slices);
              tv::cuda::Launch launcher(kernel_params.grid_dims, dim3(256),
                                          12800, reinterpret_cast<cudaStream_t>(params.stream));
              cudaError_t result;
              if (12800 >= (48 << 10)) {
                  result = cudaFuncSetAttribute(Simt_f16f16f16f32f32tnt_m32n64k32m32n32k8A1_200_S1::gemm_kernel,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  12800);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                  result = cudaFuncSetAttribute(
                      Simt_f16f16f16f32f32tnt_m32n64k32m32n32k8A1_200_S1::gemm_kernel,
                      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
              }
              {
                  tv::CUDAKernelTimerGuard timerguard("Simt_f16f16f16f32f32tnt_m32n64k32m32n32k8A1_200_S1", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                  launcher(Simt_f16f16f16f32f32tnt_m32n64k32m32n32k8A1_200_S1::gemm_kernel, kernel_params);
              }
              TV_CHECK_CUDA_ERR_V2("Simt_f16f16f16f32f32tnt_m32n64k32m32n32k8A1_200_S1", "error with params", a.shape(), b.shape(), c.shape());
              return;
            }
          }
        }
      }
    }
    if (algo_desp.tile_shape == std::array<int, 3>{32, 32, 32}){
      if (algo_desp.warp_tile_shape == std::array<int, 3>{32, 32, 8}){
        if (algo_desp.num_stage == 2 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              found = true;
              const int* a_ptr = nullptr;
              if (!a_inds.empty()){
                  a_ptr = a_inds.data_ptr<const int>();
              }
              TV_ASSERT_RT_ERR(!c_inds.empty(), "c must not empty");
              // tv::ssprint(d.ndim() == 1 ? 0 : d_ten.stride(0), (d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>()) == nullptr, "WTF");
              GemmParamsSimt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_S1 kernel_params(
                  m, n, k, a_ten.data_ptr<const tv::half_t>(), b_ten.data_ptr<const tv::half_t>(),
                  c_ten.data_ptr<tv::half_t>(), d_ten.data_ptr<tv::half_t>(), 
                  a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0), 
                  a_ptr, c_inds.data_ptr<const int>(), d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>(), 
                  float(params.alpha), float(params.beta), 
                  float(params.act_alpha), float(params.act_beta),
                  params.act_type,
                  split_k_slices);
              tv::cuda::Launch launcher(kernel_params.grid_dims, dim3(128),
                                          8704, reinterpret_cast<cudaStream_t>(params.stream));
              cudaError_t result;
              if (8704 >= (48 << 10)) {
                  result = cudaFuncSetAttribute(Simt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_S1::gemm_kernel,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  8704);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                  result = cudaFuncSetAttribute(
                      Simt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_S1::gemm_kernel,
                      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
              }
              {
                  tv::CUDAKernelTimerGuard timerguard("Simt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_S1", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                  launcher(Simt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_S1::gemm_kernel, kernel_params);
              }
              TV_CHECK_CUDA_ERR_V2("Simt_f16f16f16f32f32tnt_m32n32k32m32n32k8A1_200_S1", "error with params", a.shape(), b.shape(), c.shape());
              return;
            }
          }
        }
      }
    }
    if (algo_desp.tile_shape == std::array<int, 3>{64, 128, 16}){
      if (algo_desp.warp_tile_shape == std::array<int, 3>{32, 64, 8}){
        if (algo_desp.num_stage == 2 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              found = true;
              const int* a_ptr = nullptr;
              if (!a_inds.empty()){
                  a_ptr = a_inds.data_ptr<const int>();
              }
              TV_ASSERT_RT_ERR(!c_inds.empty(), "c must not empty");
              // tv::ssprint(d.ndim() == 1 ? 0 : d_ten.stride(0), (d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>()) == nullptr, "WTF");
              GemmParamsSimt_f16f16f16f32f32tnt_m64n128k16m32n64k8A1_200_S1 kernel_params(
                  m, n, k, a_ten.data_ptr<const tv::half_t>(), b_ten.data_ptr<const tv::half_t>(),
                  c_ten.data_ptr<tv::half_t>(), d_ten.data_ptr<tv::half_t>(), 
                  a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0), 
                  a_ptr, c_inds.data_ptr<const int>(), d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>(), 
                  float(params.alpha), float(params.beta), 
                  float(params.act_alpha), float(params.act_beta),
                  params.act_type,
                  split_k_slices);
              tv::cuda::Launch launcher(kernel_params.grid_dims, dim3(256),
                                          12800, reinterpret_cast<cudaStream_t>(params.stream));
              cudaError_t result;
              if (12800 >= (48 << 10)) {
                  result = cudaFuncSetAttribute(Simt_f16f16f16f32f32tnt_m64n128k16m32n64k8A1_200_S1::gemm_kernel,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  12800);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                  result = cudaFuncSetAttribute(
                      Simt_f16f16f16f32f32tnt_m64n128k16m32n64k8A1_200_S1::gemm_kernel,
                      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
              }
              {
                  tv::CUDAKernelTimerGuard timerguard("Simt_f16f16f16f32f32tnt_m64n128k16m32n64k8A1_200_S1", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                  launcher(Simt_f16f16f16f32f32tnt_m64n128k16m32n64k8A1_200_S1::gemm_kernel, kernel_params);
              }
              TV_CHECK_CUDA_ERR_V2("Simt_f16f16f16f32f32tnt_m64n128k16m32n64k8A1_200_S1", "error with params", a.shape(), b.shape(), c.shape());
              return;
            }
          }
        }
      }
    }
    if (algo_desp.tile_shape == std::array<int, 3>{64, 64, 8}){
      if (algo_desp.warp_tile_shape == std::array<int, 3>{32, 32, 8}){
        if (algo_desp.num_stage == 2 && algo_desp.dacc == 0 && algo_desp.dcomp == 0){
          if ((params.split_k_slices == 1)){
            if (algo_desp.access_per_vector == 1){
              found = true;
              const int* a_ptr = nullptr;
              if (!a_inds.empty()){
                  a_ptr = a_inds.data_ptr<const int>();
              }
              TV_ASSERT_RT_ERR(!c_inds.empty(), "c must not empty");
              // tv::ssprint(d.ndim() == 1 ? 0 : d_ten.stride(0), (d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>()) == nullptr, "WTF");
              GemmParamsSimt_f16f16f16f32f32tnt_m64n64k8m32n32k8A1_200_S1 kernel_params(
                  m, n, k, a_ten.data_ptr<const tv::half_t>(), b_ten.data_ptr<const tv::half_t>(),
                  c_ten.data_ptr<tv::half_t>(), d_ten.data_ptr<tv::half_t>(), 
                  a_ten.stride(0), b_ten.stride(0), c_ten.stride(0), d.ndim() == 1 ? 0 : d_ten.stride(0), 
                  a_ptr, c_inds.data_ptr<const int>(), d.ndim() == 1 ? nullptr : c_inds.data_ptr<const int>(), 
                  float(params.alpha), float(params.beta), 
                  float(params.act_alpha), float(params.act_beta),
                  params.act_type,
                  split_k_slices);
              tv::cuda::Launch launcher(kernel_params.grid_dims, dim3(128),
                                          4608, reinterpret_cast<cudaStream_t>(params.stream));
              cudaError_t result;
              if (4608 >= (48 << 10)) {
                  result = cudaFuncSetAttribute(Simt_f16f16f16f32f32tnt_m64n64k8m32n32k8A1_200_S1::gemm_kernel,
                                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                  4608);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
                  result = cudaFuncSetAttribute(
                      Simt_f16f16f16f32f32tnt_m64n64k8m32n32k8A1_200_S1::gemm_kernel,
                      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
                  TV_ASSERT_RT_ERR(result == cudaSuccess, "error");
              }
              {
                  tv::CUDAKernelTimerGuard timerguard("Simt_f16f16f16f32f32tnt_m64n64k8m32n32k8A1_200_S1", evtimer, reinterpret_cast<cudaStream_t>(params.stream));
                  launcher(Simt_f16f16f16f32f32tnt_m64n64k8m32n32k8A1_200_S1::gemm_kernel, kernel_params);
              }
              TV_CHECK_CUDA_ERR_V2("Simt_f16f16f16f32f32tnt_m64n64k8m32n32k8A1_200_S1", "error with params", a.shape(), b.shape(), c.shape());
              return;
            }
          }
        }
      }
    }
  }
  if (!found){
      TV_THROW_INVALID_ARG("Can't Found Algorithm for params:", algo_desp.tile_shape, algo_desp.warp_tile_shape, 
          algo_desp.num_stage, tv::dtype_str(a.dtype()), 
          tv::dtype_str(b.dtype()), tv::dtype_str(c.dtype()), tv::dtype_str(dacc), 
          tv::dtype_str(dcomp), ta, tb, tc, algo_desp.algo, algo_desp.tensorop);
  }
  // return 0;
}
} // namespace main
} // namespace gemm
} // namespace cumm
} // namespace spconvlib