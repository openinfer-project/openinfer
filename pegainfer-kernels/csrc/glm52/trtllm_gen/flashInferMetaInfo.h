/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <flashinfer/trtllm/fmha/kernelParams.h>

namespace tensorrt_llm {
namespace kernels {

// Minimal subset of FlashInfer 0.6.12's generated metadata needed by
// GLM5.2 TP4 decode. The checked-in cubins carry the matching SHA-256 names.
struct TllmGenFmhaKernelMetaInfo {
  Data_type mDataTypeQ;
  Data_type mDataTypeKv;
  Data_type mDataTypeK;
  Data_type mDataTypeV;
  Data_type mDataTypeO;
  int mTileSizeQ;
  int mTileSizeKv;
  int mStepQ;
  int mStepKv;
  int mHeadDimPerCtaV;
  int mHeadDimQk;
  int mHeadDimV;
  int mSM;
  const unsigned char* mCubin;
  unsigned int mCubinSize;
  const char* mFuncName;
  int mSharedMemBytes;
  int mThreadsPerCTA;
  int mQkvLayout;
  int mNumTokensPerPage;
  int mMaskType;
  int mKernelType;
  int mTileScheduler;
  int mMultiCtasKvMode;
  int mNumEltsPerSageAttnBlkQ;
  int mNumEltsPerSageAttnBlkK;
  int mNumEltsPerSageAttnBlkP;
  int mNumEltsPerSageAttnBlkV;
  bool mGroupsHeadsQ;
  bool mGroupsTokensHeadsQ;
  bool mReuseSmemKForV;
  bool m2CtaMma;
  int mSparseAttn;
  bool mSkipsSoftmaxWhenPossible;
  bool mReserved1;
  bool mReserved2;
  const char* sha256;
};

static const TllmGenFmhaKernelMetaInfo sTllmGenFmhaKernelMetaInfos[] = {
    // Selector seed: isSupported() checks the unsplit V=512 shape before the
    // launch heuristic right-sizes it to VPerCta128 on GB300.
    {DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
     DATA_TYPE_BF16, 8, 128, 8, 128, 512, 576, 512, kSM_100f, nullptr, 0,
     "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen",
     166688, 512, 2, 1, 0, 2, 0, 1, 0, 0, 0, 0, true, false, false,
     false, 1, false, false, false,
     "54ef64241e7f37e69b56cea37d4de5a79468cfbf62ac4bf87fd2b5c06fb6266a"},
    {DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
     DATA_TYPE_BF16, 8, 128, 8, 128, 512, 576, 512, kSM_100f, nullptr, 0,
     "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1VarSeqQ8Kv128PersistentSwapsAbForGen",
     168912, 512, 2, 1, 0, 2, 1, 0, 0, 0, 0, 0, true, false, false,
     false, 1, false, false, false,
     "b1cbd799fff0c586eac597d7dd2385ec6a76e2ce6dbf86eba0e691b43ebce67b"},
    {DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
     DATA_TYPE_BF16, 8, 128, 8, 128, 128, 576, 512, kSM_100f, nullptr, 0,
     "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseStaticTokenSparseP1VarSeqQ8Kv128PersistentSwapsAbForGen",
     168912, 512, 2, 1, 0, 2, 1, 0, 0, 0, 0, 0, true, false, false,
     false, 1, false, false, false,
     "77a6891a9c3837dee87d2cad5fb8d543271e063582ca78ad504b43851ba55109"},
    {DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
     DATA_TYPE_BF16, 8, 128, 8, 128, 128, 576, 512, kSM_100f, nullptr, 0,
     "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbForGen",
     166688, 512, 2, 1, 0, 2, 0, 1, 0, 0, 0, 0, true, false, false,
     false, 1, false, false, false,
     "611bdd79d0deeeb35b5600318a7591d95ae24041ad39fda04e9750b99b8854ed"},
    {DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
     DATA_TYPE_BF16, 16, 128, 16, 128, 128, 576, 512, kSM_100f, nullptr, 0,
     "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta128PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",
     179360, 512, 2, 1, 0, 2, 0, 1, 0, 0, 0, 0, true, false, false,
     false, 1, false, false, false,
     "e62c5ec93d14d10d780a5147da6982b183c7e267d65b6ee99bf057fa81c90376"},
    {DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
     DATA_TYPE_BF16, 16, 128, 16, 128, 256, 576, 512, kSM_100f, nullptr, 0,
     "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512HVPerCta256PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",
     179360, 512, 2, 1, 0, 2, 0, 1, 0, 0, 0, 0, true, false, false,
     false, 1, false, false, false,
     "97b699c634f7f56b72cedfa99214c89b955edcce2577cf830958e953d4f8d4e7"},
    {DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3, DATA_TYPE_E4M3,
     DATA_TYPE_BF16, 16, 128, 16, 128, 512, 576, 512, kSM_100f, nullptr, 0,
     "fmhaSm100fKernel_QkvE4m3OBfloat16HQk576HV512PagedKvDenseStaticTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",
     179360, 512, 2, 1, 0, 2, 0, 1, 0, 0, 0, 0, true, false, false,
     false, 1, false, false, false,
     "c567e388756b51b7f732e8ce4c9627f46496fb75bee8393ee6adfd9eb57ae312"},
};

}  // namespace kernels
}  // namespace tensorrt_llm
