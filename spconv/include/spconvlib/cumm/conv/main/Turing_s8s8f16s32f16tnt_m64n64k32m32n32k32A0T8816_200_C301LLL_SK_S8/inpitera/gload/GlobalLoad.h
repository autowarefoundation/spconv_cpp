#pragma once
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Turing_s8s8f16s32f16tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8 {
namespace inpitera {
namespace gload {
struct GlobalLoad {
  template <typename Frag>
  __forceinline__ __device__ static void run(Frag & frag, void const* ptr, bool pred)   {
    
    uint8_t* frag_ptr = reinterpret_cast<uint8_t*>(&frag);
    if (pred){
        frag_ptr[0] = reinterpret_cast<uint8_t const*>(ptr)[0];
    }
  }
};
} // namespace gload
} // namespace inpitera
} // namespace Turing_s8s8f16s32f16tnt_m64n64k32m32n32k32A0T8816_200_C301LLL_SK_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib