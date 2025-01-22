#pragma once
#include <spconvlib/cumm/common/TensorViewNVRTCKernel.h>
#include <spconvlib/cumm/gemm/layout/RowMajor.h>
#include <spconvlib/cumm/gemm/layout/ColumnMajor.h>
#include <spconvlib/cumm/common/GemmBasicKernel.h>
#include <spconvlib/cumm/common/GemmBasic.h>
#include <spconvlib/cumm/common/GemmKernelFlags.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/inpitera/ForwardDgradSparseIOIterator.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/inpiterb/WeightIteratorDP4A.h>
#include <spconvlib/cumm/conv/main/cpAmpere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/la/TensorGeneric.h>
#include <spconvlib/cumm/conv/main/cpAmpere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/lb/TensorGeneric.h>
#include <spconvlib/cumm/conv/main/cpAmpere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/lc/TensorGeneric.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/gemm_smem_storage/BlockMmaStorage.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/out_smem_storage/OutputSmemStorage.h>
#include <spconvlib/cumm/conv/main/cpAmpere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/ConvParams.h>
#include <spconvlib/cumm/conv/main/cpAmpere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/cp/ConvProblem.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/out_iter/OutIterator.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/out_iter_const/OutIterator.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/scale_out_iter_const/OutIterator.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/out_op/Int8Inference.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/mma/MmaMultiStage.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/mma_miterd/MaskIGemmIteratorMaskLoaderDynamic.h>
#include <spconvlib/cumm/conv/main/Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8/output/Output.h>
namespace spconvlib {
namespace cumm {
namespace conv {
namespace main {
namespace Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8 {
using TensorViewNVRTCKernel = spconvlib::cumm::common::TensorViewNVRTCKernel;
using RowMajor = spconvlib::cumm::gemm::layout::RowMajor;
using ColumnMajor = spconvlib::cumm::gemm::layout::ColumnMajor;
using GemmBasicKernel = spconvlib::cumm::common::GemmBasicKernel;
using GemmBasic = spconvlib::cumm::common::GemmBasic;
using GemmKernelFlags = spconvlib::cumm::common::GemmKernelFlags;
using InputIteratorA = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::inpitera::ForwardDgradSparseIOIterator;
using InputIteratorB = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::inpiterb::WeightIteratorDP4A;
using LayoutA = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::la::TensorGeneric;
using LayoutB = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::lb::TensorGeneric;
using LayoutC = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::lc::TensorGeneric;
using BlockMmaStorage = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::gemm_smem_storage::BlockMmaStorage;
using OutputSmemStorage = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::out_smem_storage::OutputSmemStorage;
using ConvParams = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::ConvParams;
using ConvProblem = spconvlib::cumm::conv::main::cpAmpere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::cp::ConvProblem;
using OutIter = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::out_iter::OutIterator;
using ConstOutIter = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::out_iter_const::OutIterator;
using ConstScaleOutIter = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::scale_out_iter_const::OutIterator;
using OutputOp = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::out_op::Int8Inference;
using Mma = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::mma::MmaMultiStage;
using MaskIGemmIteratorDynamic = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::mma_miterd::MaskIGemmIteratorMaskLoaderDynamic;
using Output = spconvlib::cumm::conv::main::Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8::output::Output;
/**
 * @param params 
 */
__global__ void conv_kernel(ConvParams params);
struct ConvKernel {
};
} // namespace Ampere_s8s8s8s32f16tnt_m128n256k64m64n128k64A1T16832_300_C301LLL_SKD_S8
} // namespace main
} // namespace conv
} // namespace cumm
} // namespace spconvlib