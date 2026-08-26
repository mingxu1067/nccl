/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"

template <typename T, typename RedOp>
struct RunWorkBatch<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  static_assert(sizeof(T) == 1, "SendRecv only works on single byte types T.");

  template <int NPeers>
  struct A2aFusedStep {
    const __nv_bfloat16* local;
    __nv_bfloat16* output;
    int cursor;
    int remaining;
    ncclDevWorkColl* work;

    template <int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int MultimemSrcs,
              int MultimemDsts>
    __device__ __forceinline__ void operator()(int tid, int nthreads, int slice, int maxSliceSize, int nSrcs,
                                               void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes,
                                               uint32_t sendDirectFlag, uint32_t recvDirectFlag) {
      static_assert(SlicePerChunk == 1, "A2AFused requires one Simple slice per step");
      const int nelem = min(maxSliceSize, remaining - cursor);
      ncclA2aFusedReducePtrs<NPeers>(tid, nthreads, local + cursor, srcPtrs, output + cursor, nelem);
    }
  };

  template <int NPeers>
  struct A2aFusedAccumStep {
    const __nv_bfloat16* local;
    float* accum;
    __nv_bfloat16* output;
    int cursor;
    int remaining;
    bool firstWindow;
    bool lastWindow;
    ncclDevWorkColl* work;

    template <int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int MultimemSrcs,
              int MultimemDsts>
    __device__ __forceinline__ void operator()(int tid, int nthreads, int slice, int maxSliceSize, int nSrcs,
                                               void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes,
                                               uint32_t sendDirectFlag, uint32_t recvDirectFlag) {
      static_assert(SlicePerChunk == 1, "A2AFused requires one Simple slice per step");
      const int nelem = min(maxSliceSize, remaining - cursor);
      if (firstWindow) {
        if (lastWindow)
          ncclA2aFusedAccumulatePtrs<NPeers, true, true>(tid, nthreads, local + cursor, srcPtrs, accum + cursor,
                                                         output + cursor, nelem);
        else
          ncclA2aFusedAccumulatePtrs<NPeers, true, false>(tid, nthreads, local + cursor, srcPtrs, accum + cursor,
                                                          output + cursor, nelem);
      } else {
        if (lastWindow)
          ncclA2aFusedAccumulatePtrs<NPeers, false, true>(tid, nthreads, local + cursor, srcPtrs, accum + cursor,
                                                          output + cursor, nelem);
        else
          ncclA2aFusedAccumulatePtrs<NPeers, false, false>(tid, nthreads, local + cursor, srcPtrs, accum + cursor,
                                                           output + cursor, nelem);
      }
    }
  };

  template <int NPeers>
  __device__ __forceinline__ void runA2aFusedPeers(int tid, int nthreads, struct ncclDevWorkP2p* works) {
    static_assert(0 < NPeers && NPeers <= NCCL_MAX_DIRECT_ARITY, "Unsupported A2AFused peer count");
    int recvPeers[NPeers];
    #pragma unroll
    for (int peer = 0; peer < NPeers; peer++) recvPeers[peer] = works[peer].recvRank;

    const int part = ncclP2pChannelToPart(works[0].nP2pChannels, works[0].channelBase, ncclShmem.channelId);
    size_t byteBegin, byteEnd;
    ncclP2pPartBounds(works[0].nSendChannels, part, works[0].a2aRecvcount * sizeof(__nv_bfloat16), &byteBegin,
                      &byteEnd);
    const int partCount = (byteEnd - byteBegin) / sizeof(__nv_bfloat16);
    const auto* local = static_cast<const __nv_bfloat16*>(works[0].a2aInput) +
                        size_t(ncclShmem.comm.rank) * works[0].a2aRecvcount + byteBegin / sizeof(__nv_bfloat16);
    auto* output = static_cast<__nv_bfloat16*>(works[0].a2aOutput) + byteBegin / sizeof(__nv_bfloat16);

    // Match NCCL SendRecv's deadlock-free execution model: peer-specific send
    // groups run concurrently with one multi-peer owner receive/reduction
    // group, all inside this CTA.
    // One Simple send warp per peer leaves the maximum number of workers for
    // the owner receive/reduction side, which is the limiting stage. This also
    // leaves receive workers at every supported rank count.
    constexpr int sendThreads = NPeers * WARP_SIZE;
    if (tid < sendThreads) {
      const int work = tid / WARP_SIZE;
      const int sendTid = tid - work * WARP_SIZE;
      runSend<ProtoSimple<1, 1>>(sendTid, WARP_SIZE, work + 1, &works[work]);
      return;
    }

    const int recvTid = tid - sendThreads;
    const int recvThreads = nthreads - sendThreads;
    using Proto = ProtoSimple<1, 1>;
    Primitives<__nv_bfloat16, FuncSum<__nv_bfloat16>, FanAsymmetric<NPeers, 0>, 0, Proto, 1> prims(
        recvTid, recvThreads, recvPeers, nullptr, nullptr, nullptr, 0, 0, 1, 1, nullptr, &works[0],
        ncclShmem.comm.p2pChunkSize / sizeof(__nv_bfloat16));
    const int stepElements = ncclShmem.comm.p2pChunkSize / sizeof(__nv_bfloat16);
    for (int cursor = 0; cursor < partCount; cursor += stepElements) {
      A2aFusedStep<NPeers> step{local, output, cursor, partCount, nullptr};
      prims.template process<1, 0>(step);
    }
  }

  template <int NPeers>
  __device__ __forceinline__ void runA2aFusedWindow(int tid, int nthreads, struct ncclDevWorkP2p* works,
                                                    const __nv_bfloat16* local, float* accum,
                                                    __nv_bfloat16* output, int partCount, bool firstWindow,
                                                    bool lastWindow) {
    static_assert(0 < NPeers && NPeers <= NCCL_MAX_DIRECT_ARITY, "Unsupported A2AFused peer window");
    // Keep a fixed seven send-warps layout for every window so the receiver
    // worker IDs and Simple barrier groups do not change between windows.
    constexpr int sendThreads = NCCL_MAX_DIRECT_ARITY * WARP_SIZE;
    if (tid < sendThreads) {
      const int work = tid / WARP_SIZE;
      if (work < NPeers) runSend<ProtoSimple<1, 1>>(tid - work * WARP_SIZE, WARP_SIZE, work + 1, &works[work]);
    } else {
      int recvPeers[NPeers];
      #pragma unroll
      for (int peer = 0; peer < NPeers; peer++) recvPeers[peer] = works[peer].recvRank;
      const int recvTid = tid - sendThreads;
      const int recvThreads = nthreads - sendThreads;
      using Proto = ProtoSimple<1, 1>;
      Primitives<__nv_bfloat16, FuncSum<__nv_bfloat16>, FanAsymmetric<NPeers, 0>, 0, Proto, 1> prims(
          recvTid, recvThreads, recvPeers, nullptr, nullptr, nullptr, 0, 0, 1, 1, nullptr, &works[0],
          ncclShmem.comm.p2pChunkSize / sizeof(__nv_bfloat16));
      const int stepElements = ncclShmem.comm.p2pChunkSize / sizeof(__nv_bfloat16);
      for (int cursor = 0; cursor < partCount; cursor += stepElements) {
        A2aFusedAccumStep<NPeers> step{local, accum, output, cursor, partCount, firstWindow, lastWindow, nullptr};
        prims.template process<1, 0>(step);
      }
    }
  }

  __device__ __forceinline__ void runA2aFusedWindows(int tid, int nthreads, struct ncclDevWorkP2p* works,
                                                     int nWorks) {
    const int part = ncclP2pChannelToPart(works[0].nP2pChannels, works[0].channelBase, ncclShmem.channelId);
    size_t byteBegin, byteEnd;
    ncclP2pPartBounds(works[0].nSendChannels, part, works[0].a2aRecvcount * sizeof(__nv_bfloat16), &byteBegin,
                      &byteEnd);
    const int partCount = (byteEnd - byteBegin) / sizeof(__nv_bfloat16);
    const auto* local = static_cast<const __nv_bfloat16*>(works[0].a2aInput) +
                        size_t(ncclShmem.comm.rank) * works[0].a2aRecvcount + byteBegin / sizeof(__nv_bfloat16);
    auto* output = static_cast<__nv_bfloat16*>(works[0].a2aOutput) + byteBegin / sizeof(__nv_bfloat16);
    // recvAddr points at scratch[recvRank][byteBegin] after channel
    // partitioning. Recover the scratch base without growing every peer work
    // descriptor with another pointer.
    auto* scratchBase = static_cast<char*>(works[0].recvAddr) -
                        size_t(works[0].recvRank) * works[0].a2aRecvcount * sizeof(__nv_bfloat16) - byteBegin;
    auto* accum = reinterpret_cast<float*>(scratchBase) + byteBegin / sizeof(__nv_bfloat16);

    for (int base = 0; base < nWorks; base += NCCL_MAX_DIRECT_ARITY) {
      const int peers = min(NCCL_MAX_DIRECT_ARITY, nWorks - base);
      const bool firstWindow = base == 0;
      const bool lastWindow = base + peers == nWorks;
      switch (peers) {
        case 1: runA2aFusedWindow<1>(tid, nthreads, works + base, local, accum, output, partCount, firstWindow,
                                     lastWindow); break;
        case 2: runA2aFusedWindow<2>(tid, nthreads, works + base, local, accum, output, partCount, firstWindow,
                                     lastWindow); break;
        case 3: runA2aFusedWindow<3>(tid, nthreads, works + base, local, accum, output, partCount, firstWindow,
                                     lastWindow); break;
        case 4: runA2aFusedWindow<4>(tid, nthreads, works + base, local, accum, output, partCount, firstWindow,
                                     lastWindow); break;
        case 5: runA2aFusedWindow<5>(tid, nthreads, works + base, local, accum, output, partCount, firstWindow,
                                     lastWindow); break;
        case 6: runA2aFusedWindow<6>(tid, nthreads, works + base, local, accum, output, partCount, firstWindow,
                                     lastWindow); break;
        case 7: runA2aFusedWindow<7>(tid, nthreads, works + base, local, accum, output, partCount, firstWindow,
                                     lastWindow); break;
      }
      __syncthreads();
    }
  }

  __device__ __forceinline__ void runA2aFused(int tid, int nthreads, struct ncclDevWorkP2p* works, int nWorks) {
    if (nWorks > NCCL_MAX_DIRECT_ARITY) return runA2aFusedWindows(tid, nthreads, works, nWorks);
    switch (nWorks) {
      case 1: return runA2aFusedPeers<1>(tid, nthreads, works);
      case 2: return runA2aFusedPeers<2>(tid, nthreads, works);
      case 3: return runA2aFusedPeers<3>(tid, nthreads, works);
      case 4: return runA2aFusedPeers<4>(tid, nthreads, works);
      case 5: return runA2aFusedPeers<5>(tid, nthreads, works);
      case 6: return runA2aFusedPeers<6>(tid, nthreads, works);
      case 7: return runA2aFusedPeers<7>(tid, nthreads, works);
    }
  }

  template <typename Proto>
  __device__ void runSend(int tid, int tn, int group, struct ncclDevWorkP2p* work) {
    size_t bytes = work->sendBytes;
    bool useLargeChunk = (work->sendIpcReg && ncclShmem.comm.isAllNvlink) || work->sendNetReg;
    int chunkSize = useLargeChunk ? NCCL_MAX_NET_SIZE : u32fp8Decode(work->sendChunkSize_u32fp8);
    int stepSize = useLargeChunk ? NCCL_MAX_NET_SIZE : ncclShmem.comm.p2pChunkSize;
    Primitives<T, RedOp, FanAsymmetric<0, 1>, 1, Proto, 1> prims(tid, tn, nullptr, &work->sendRank, work->sendAddr,
                                                                 nullptr,
                                                                 /*redOpArg(ignored)=*/0, group, 1, 1, nullptr, work,
                                                                 stepSize);
    size_t cursor = 0;
    do {
      int n = min(size_t(chunkSize), bytes - cursor);
      prims.directSend(cursor, cursor, n);
      cursor += n;
    } while (cursor < bytes);
  }

  template <typename Proto>
  __device__ void runRecv(int tid, int tn, int group, struct ncclDevWorkP2p* work) {
    size_t bytes = work->recvBytes;
    bool useLargeChunk = (work->recvIpcReg && ncclShmem.comm.isAllNvlink) || work->recvNetReg;
    int chunkSize = useLargeChunk ? NCCL_MAX_NET_SIZE : u32fp8Decode(work->recvChunkSize_u32fp8);
    int stepSize = useLargeChunk ? NCCL_MAX_NET_SIZE : ncclShmem.comm.p2pChunkSize;
    Primitives<T, RedOp, FanAsymmetric<1, 0>, 1, Proto, 1> prims(tid, tn, &work->recvRank, nullptr, nullptr,
                                                                 work->recvAddr,
                                                                 /*redOpArg(ignored)=*/0, group, 1, 1, nullptr, work,
                                                                 stepSize);
    size_t cursor = 0;
    do {
      int n = min(size_t(chunkSize), bytes - cursor);
      prims.directRecv(cursor, n);
      cursor += n;
    } while (cursor < bytes);
  }

  __device__ __forceinline__ void run() {
    const int tid = threadIdx.x;
    const int tn = blockDim.x;
    const int wid = tid / WARP_SIZE;
    const int nWarps = tn / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    struct Shared {
      uint32_t workSendMask; // bitmasks of which work indices have send/recv
      uint32_t workRecvMask;
    };
    Shared* shared = (Shared*)ncclScratchForWarp(0);

    struct ncclDevWorkP2p* works = (ncclDevWorkP2p*)ncclShmem.workStorage;
    int nWorks = ncclShmem.nWorks;

    if (works[0].a2aFused && nWorks > NCCL_MAX_DIRECT_ARITY) {
      // A2AFused extension batches may contain as many as 127 descriptors.
      // Partition every descriptor before entering the peer-window loop.
      for (int item = tid; item < 2 * nWorks; item += tn) {
        int workIx = item / 2;
        int isSend = item & 1;
        struct ncclDevWorkP2p* work = &works[workIx];
        size_t bytes = isSend ? work->sendBytes : work->recvBytes;
        int nParts = isSend ? work->nSendChannels : work->nRecvChannels;
        int part = ncclP2pChannelToPart(work->nP2pChannels, work->channelBase, ncclShmem.channelId);
        if (nParts != 0) {
          size_t partBeg, partEnd;
          ncclP2pPartBounds(nParts, part, bytes, &partBeg, &partEnd);
          (isSend ? work->sendAddr : work->recvAddr) =
              static_cast<char*>(isSend ? work->sendAddr : work->recvAddr) + partBeg;
          (isSend ? work->sendBytes : work->recvBytes) = partEnd - partBeg;
        }
      }
      __syncthreads();
      runA2aFused(tid, tn, works, nWorks);
      return;
    }

    if (wid == 0) {
      // Modify the memory range of each work[] to reflect this channel's
      // partition of the work. Since integer divides are very heavy it's
      // best to do them all in one warp.
      int workIx = lane % 16;
      int isSend = lane < 16 ? 0 : 1;
      bool hasWork = false;
      if (workIx < nWorks) {
        struct ncclDevWorkP2p* work = &works[workIx];
        size_t bytes = isSend ? work->sendBytes : work->recvBytes;
        int nParts = isSend ? work->nSendChannels : work->nRecvChannels;
        int part = ncclP2pChannelToPart(work->nP2pChannels, work->channelBase, ncclShmem.channelId);
        hasWork = (part < nParts);
        if (nParts != 0) {
          size_t partBeg, partEnd;
          ncclP2pPartBounds(nParts, part, bytes, &partBeg, &partEnd);
          (isSend ? work->sendAddr : work->recvAddr) = (char*)(isSend ? work->sendAddr : work->recvAddr) + partBeg;
          (isSend ? work->sendBytes : work->recvBytes) = partEnd - partBeg;
        }
      }
      // Coverity reports a possible thread divergence due to not all threads participating in the collective.
      // However, the code ensures that the participation is on a per-warp basis.
      // coverity[device_thread_diverged:FALSE]
      uint32_t mask = __ballot_sync(~0u, hasWork);
      if (lane == 0) {
        shared->workSendMask = mask >> 16;
        shared->workRecvMask = mask & 0xffff;
      }
    }

    // The fastest way to compute a warp uniform division x/y in [0,32) is to
    // use each lane to guess a solution and count the ones that don't exceed
    // the numerator:
    //   __popc(__ballot_sync(~0u, y*(lane+1) <= x))
    // That takes 1/3 the time of standard division and about 3/4 the time of
    // approximate floating point division:
    //   __float2int_rd(__fdividef(float(x),float(y))).

    // nWarpPerWork = nWarps/nWorks
    int nWarpPerWork = __popc(__ballot_sync(~0u, nWorks * (lane + 1) <= nWarps));
    int nRecvWarpPerWork = nWarpPerWork <= 4 ? nWarpPerWork / 2 : (nWarpPerWork - 1) / 2;
    int nSendWarpPerWork = nWarpPerWork <= 4 ? nRecvWarpPerWork : nRecvWarpPerWork + 1;
    // This might reduce nWarpPerWork which is probably desirable. It is better
    // to have a balanced number of reading and writing threads even if that
    // leaves warps unused.
    nWarpPerWork = nSendWarpPerWork + nRecvWarpPerWork;
    // The work index this warp belongs to: workIx = wid/nWarpPerWork
    int workIx = __popc(__ballot_sync(~0u, (lane + 1) * nWarpPerWork <= wid));

    __syncthreads(); // Wait for works[] and shared->* to be updated by warp=0

    uint32_t workSendMask = shared->workSendMask;
    uint32_t workRecvMask = shared->workRecvMask;

    __syncthreads(); // release scratch space used by shared->*
    if (works[0].a2aFused) {
      runA2aFused(tid, tn, works, nWorks);
      return;
    }
    if (nWorks <= workIx) return;

    // Thread range for whole work (send & recv combined)
    int subtid = tid - workIx * nWarpPerWork * WARP_SIZE;
    int subtn = nWarpPerWork * WARP_SIZE;

    // A send primtive of sufficient size requires 2 cuda barrier ids.
    constexpr int nSendWarpsForExtraGroup = NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE / WARP_SIZE;
    // Count up all group ids used below this workIx:
    int group, extra;
    // Each recv gets one group id:
    group = __popc(workRecvMask & ((1 << workIx) - 1));
    // Sends accompanying recvs get one and maybe an extra:
    extra = (nSendWarpPerWork >= nSendWarpsForExtraGroup) ? 1 : 0;
    group += __popc((workSendMask & workRecvMask) & ((1 << workIx) - 1)) * (1 + extra);
    // Sends without recvs use more warps so compute extra accordingly:
    extra = (nWarpPerWork >= nSendWarpsForExtraGroup) ? 1 : 0;
    group += __popc((workSendMask & ~workRecvMask) & ((1 << workIx) - 1)) * (1 + extra);

    struct ncclDevWorkP2p* work = &works[workIx];
    bool hasSend = 1 & (workSendMask >> workIx);
    bool hasRecv = 1 & (workRecvMask >> workIx);
    bool isCopy = work->sendRank == ncclShmem.comm.rank;
    bool isSend = !hasRecv || (hasSend && subtid < nSendWarpPerWork * WARP_SIZE);

    if (!isCopy && hasSend && hasRecv) {
      // Translate thread ids to reflect just this send or recv as opposed to whole work.
      if (isSend) {
        subtn = nSendWarpPerWork * WARP_SIZE;
      } else {
        subtid -= nSendWarpPerWork * WARP_SIZE;
        subtn = nRecvWarpPerWork * WARP_SIZE;
        group += 1 + (nSendWarpPerWork >= nSendWarpsForExtraGroup ? 1 : 0);
      }
    }

    if (isCopy) {
      reduceCopy<COLL_UNROLL, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs=*/0>(
        subtid, subtn, 0, false, 1, &work->sendAddr, 1, &work->recvAddr, (ssize_t)work->sendBytes);
    } else if (isSend) {
      if (work->sendProtoLL) {
        runSend<ProtoLL>(subtid, subtn, group, work);
      } else {
        runSend<ProtoSimple<1, 1>>(subtid, subtn, group, work);
      }
    } else {
      if (work->recvProtoLL) {
        runRecv<ProtoLL>(subtid, subtn, group, work);
      } else {
        runRecv<ProtoSimple<1, 1>>(subtid, subtn, group, work);
      }
    }
  }
};
