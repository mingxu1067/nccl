/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "collectives.h"
#include "core.h"
#include "utils.h"

// Used to pass NCCL call information between functions
struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root; // peer for p2p operations
  ncclComm_t comm;
  cudaStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;
  // One-sided ops
  size_t peerWinOffset;
  ncclWindow_t peerWin;
  int sigIdx;
  int ctx;
  unsigned int flags;
  // WaitSignal descriptors
  int nDesc;
  ncclWaitSignalDesc_t* signalDescs;
  // Experimental fused BF16-I/O / FP32-Ring-Simple accumulation metadata.
  void* accScratch;
  size_t accScratchBytes;
  size_t accCount;
  uint8_t mixedPrecision;
  uint8_t a2aMP;
};

#endif
