/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common_kernel.h"
#include "common.h"

#define NCCL_SPINS_BEFORE_CHECK_ABORT 10000

/* Protocol classes: ProtoSimple, ProtoLL, ProtoLL128
 * We use these as template args to the Primtiives class instead of integral
 * enums (e.g. NCCL_PROTO_LL) because for SIMPLE we need to carry a few extra
 * numbers. Also these types hold methods which let us compute numbers important
 * to how that protocol operates with a consistent interface so that our
 * algorithm code can operate protocol parametrically.
 */
template <int SlicePerChunk_1, int StepPerSlice_1, int Unroll_1 = COLL_UNROLL, int MultimemSrcs_1 = 0,
          int MultimemDsts_1 = 0>
struct ProtoSimple {
  static constexpr int Id = NCCL_PROTO_SIMPLE;
  static constexpr int SlicePerChunk = SlicePerChunk_1;
  static constexpr int StepPerSlice = StepPerSlice_1;
  static constexpr int Unroll = Unroll_1;
  static constexpr int MultimemSrcs = MultimemSrcs_1;
  static constexpr int MultimemDsts = MultimemDsts_1;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS;
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // Bogus value? Nobody queries this metric for simple.
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 2;
};

struct ProtoLL {
  static constexpr int Id = NCCL_PROTO_LL;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return ncclShmem.comm.buffSizes[NCCL_PROTO_LL] / NCCL_STEPS / 2; // Half is data
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return sizeof(uint64_t); // One 16-byte line has 8-bytes of data
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 1;
};

struct ProtoLL128 {
  static constexpr int Id = NCCL_PROTO_LL128;

  // Data bytes (no flags etc) in one step of the fifo queue.
  __device__ static int calcBytePerStep() {
    return (ncclShmem.comm.buffSizes[NCCL_PROTO_LL128] / NCCL_STEPS) * NCCL_LL128_DATAELEMS / NCCL_LL128_LINEELEMS;
  }
  // Granularity of data bytes transferred per thread.
  __device__ static int calcBytePerGrain() {
    return NCCL_LL128_SHMEM_ELEMS_PER_THREAD * NCCL_LL128_DATAELEMS * sizeof(uint64_t) / NCCL_LL128_LINEELEMS;
  }
  // Group width is how many consecutive group values a subchannel occupies.
  static constexpr int MaxGroupWidth = 1;
};

/* Fan (as in fan-in & fan-out) classes hold recv and send counts. The template
 * arguments are static bounds on the maximum values. Asymmetric counts are
 * independent. Symmetric is a static guarantee that nrecv==nsend, so it only
 * stores one value at runtime. This optimization save 32-bit register, but more
 * importantly uses fewer predicate registers when unrolling loops.
 */
template <int MaxRecv_, int MaxSend_>
struct FanAsymmetric {
  static constexpr int MaxRecv = MaxRecv_, MaxSend = MaxSend_;
  int nr, ns;
  FanAsymmetric() = default;
  __device__ FanAsymmetric(int nrecv, int nsend) : nr(nrecv), ns(nsend) {
    // assert(nrecv <= MaxRecv && nsend <= MaxSend);
  }
  __device__ int nrecv() const {
    return MaxRecv ? nr : 0;
  }
  __device__ int nsend() const {
    return MaxSend ? ns : 0;
  }
};

template <int MaxArity>
struct FanSymmetric {
  static constexpr int MaxRecv = MaxArity, MaxSend = MaxArity;
  int n;
  FanSymmetric() = default;
  __device__ FanSymmetric(int nrecv, int nsend) : n(nrecv) {
    // assert(nrecv == nsend && nrecv <= MaxArity);
  }
  __device__ int nrecv() const {
    return n;
  }
  __device__ int nsend() const {
    return n;
  }
};

// The primitives class. Specialized per protocol in the other headers.
template <typename T, typename RedOp, typename Fan, int Direct, typename Proto, int P2p, bool isNetOffload = false>
class Primitives;

// Used by LL & LL128 to implement direct members in the naive way.
template <typename RealPrimitives>
struct PrimitivesWithoutDirect {
  __device__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->send(inpIx, eltN);
  }
  __device__ void directSendFromOutput(intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->sendFromOutput(outIx, eltN);
  }
  __device__ void directRecv(intptr_t outIx, int eltN) {
    static_cast<RealPrimitives*>(this)->recv(outIx, eltN, /*postOp=*/false);
  }
  __device__ void directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    static_cast<RealPrimitives*>(this)->copySend(inpIx, outIx, eltN, postOp);
  }
  __device__ void directRecvCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    static_cast<RealPrimitives*>(this)->recvCopySend(outIx, eltN, /*postOp=*/false);
  }
  __device__ void directRecvDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    return;
  }
  __device__ void recvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp = false) {
    // Direct is only for the send part
    static_cast<RealPrimitives*>(this)->recvReduceCopySend(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN,
                                                             bool postOp = false) {
    static_cast<RealPrimitives*>(this)->recvReduceSend(inpIx, eltN);
  }
  __device__ __forceinline__ void directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN,
                                                                 bool postOp = false) {
    static_cast<RealPrimitives*>(this)->recvReduceCopySend(inpIx, outIx, eltN, postOp);
  }
};

__device__ inline int checkAbort(int& abortCache, const int abortValue, int& spins) {
  if (abortCache & abortValue) return 1;
  if (++spins < NCCL_SPINS_BEFORE_CHECK_ABORT) return 0;
  spins = 0;
  int abort = *ncclShmem.comm.abortFlag;
  if (abort) {
    ncclShmem.aborted = abort;
    abortCache |= abortValue;
  }
  return abort;
}

#include "prims_simple.h"
#include "prims_ll.h"
#include "prims_ll128.h"

template <int NPeers, bool FirstWindow, bool LastWindow>
__device__ __forceinline__ void ncclA2aMPAccumulatePtrs(int tid, int nthreads, const __nv_bfloat16* local,
                                                           void* const* remotes, float* accum,
                                                           __nv_bfloat16* output, int nelem) {
  static_assert(0 < NPeers && NPeers <= NCCL_MAX_DIRECT_ARITY, "Unsupported A2AMP peer count");
  union alignas(16) Bf16Pack8 { BytePack<16> bytes; __nv_bfloat162 b[4]; };
  union alignas(16) FloatPack4 { BytePack<16> bytes; float f[4]; };
  // Two packed loads give the best measured balance between instruction-level
  // parallelism and the length of each thread's FP32 dependency chain.
  constexpr int ElementsPerThread = 16;
  constexpr int PacksPerThread = ElementsPerThread / 8;
  const int nPacks = nelem / ElementsPerThread;
  for (int i = tid; i < nPacks; i += nthreads) {
    float2 sum[ElementsPerThread / 2];
    if (FirstWindow) {
      #pragma unroll
      for (int pack = 0; pack < PacksPerThread; pack++) {
        Bf16Pack8 value;
        value.bytes = ld_global<16>(cvta_to_global(local) + 16 * (PacksPerThread * i + pack));
        #pragma unroll
        for (int pair = 0; pair < 4; pair++) sum[4 * pack + pair] = __bfloat1622float2(value.b[pair]);
      }
    } else {
      #pragma unroll
      for (int pack = 0; pack < ElementsPerThread / 4; pack++) {
        FloatPack4 value;
        value.bytes = ld_global<16>(cvta_to_global(accum) + 16 * (4 * i + pack));
        sum[2 * pack + 0] = make_float2(value.f[0], value.f[1]);
        sum[2 * pack + 1] = make_float2(value.f[2], value.f[3]);
      }
    }
    #pragma unroll
    for (int peer = 0; peer < NPeers; peer++) {
      const auto* remote = static_cast<const __nv_bfloat16*>(remotes[peer]);
      #pragma unroll
      for (int pack = 0; pack < PacksPerThread; pack++) {
        Bf16Pack8 value;
        value.bytes = ld_volatile_global<16>(cvta_to_global(remote) + 16 * (PacksPerThread * i + pack));
        #pragma unroll
        for (int pair = 0; pair < 4; pair++) {
          const float2 addend = __bfloat1622float2(value.b[pair]);
          sum[4 * pack + pair].x += addend.x;
          sum[4 * pack + pair].y += addend.y;
        }
      }
    }
    if (LastWindow) {
      #pragma unroll
      for (int pack = 0; pack < PacksPerThread; pack++) {
        Bf16Pack8 value;
        #pragma unroll
        for (int pair = 0; pair < 4; pair++)
          value.b[pair] = __floats2bfloat162_rn(sum[4 * pack + pair].x, sum[4 * pack + pair].y);
        st_global<16>(cvta_to_global(output) + 16 * (PacksPerThread * i + pack), value.bytes);
      }
    } else {
      #pragma unroll
      for (int pack = 0; pack < ElementsPerThread / 4; pack++) {
        FloatPack4 value;
        value.f[0] = sum[2 * pack + 0].x;
        value.f[1] = sum[2 * pack + 0].y;
        value.f[2] = sum[2 * pack + 1].x;
        value.f[3] = sum[2 * pack + 1].y;
        st_global<16>(cvta_to_global(accum) + 16 * (4 * i + pack), value.bytes);
      }
    }
  }
  for (int i = nPacks * ElementsPerThread + tid; i < nelem; i += nthreads) {
    float sum = FirstWindow ? __bfloat162float(local[i]) : accum[i];
    #pragma unroll
    for (int peer = 0; peer < NPeers; peer++) {
      union { BytePack<2> bytes; __nv_bfloat16 value; } remote;
      remote.bytes = ld_volatile_global<2>(cvta_to_global(static_cast<const __nv_bfloat16*>(remotes[peer]) + i));
      sum += __bfloat162float(remote.value);
    }
    if (LastWindow) output[i] = __float2bfloat16(sum);
    else accum[i] = sum;
  }
}

template <int NPeers>
__device__ __forceinline__ void ncclA2aMPReducePtrs(int tid, int nthreads, const __nv_bfloat16* local,
                                                       void* const* remotes, __nv_bfloat16* output, int nelem) {
  ncclA2aMPAccumulatePtrs<NPeers, true, true>(tid, nthreads, local, remotes, nullptr, output, nelem);
}

#endif
