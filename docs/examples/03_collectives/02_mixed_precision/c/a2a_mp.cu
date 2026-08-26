/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for license information.
 *************************************************************************/

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define MPI_CHECK(call) do { if ((call) != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1); } while (0)
#define CUDA_CHECK(call) do { \
  cudaError_t status = (call); \
  if (status != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)
#define NCCL_CHECK(call) do { \
  ncclResult_t status = (call); \
  if (status != ncclSuccess) { \
    std::fprintf(stderr, "NCCL error: %s\n", ncclGetErrorString(status)); \
    MPI_Abort(MPI_COMM_WORLD, 1); \
  } \
} while (0)

int main(int argc, char** argv) {
  MPI_CHECK(MPI_Init(&argc, &argv));
  int rank, nranks;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
  CUDA_CHECK(cudaSetDevice(rank));

  ncclUniqueId id;
  if (rank == 0) NCCL_CHECK(ncclGetUniqueId(&id));
  MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  ncclComm_t comm;
  NCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

  constexpr size_t recvcount = 4096;
  const size_t count = recvcount * nranks;
  const size_t scratchBytes = count * sizeof(__nv_bfloat16);
  std::vector<__nv_bfloat16> input(count, __float2bfloat16(float(rank + 1)));
  __nv_bfloat16 *send, *allReduceOutput, *reduceScatterOutput, *scratch;
  CUDA_CHECK(cudaMalloc(&send, count * sizeof(*send)));
  CUDA_CHECK(cudaMalloc(&allReduceOutput, count * sizeof(*allReduceOutput)));
  CUDA_CHECK(cudaMalloc(&reduceScatterOutput, recvcount * sizeof(*reduceScatterOutput)));
  CUDA_CHECK(cudaMalloc(&scratch, scratchBytes));
  CUDA_CHECK(cudaMemcpy(send, input.data(), count * sizeof(*send), cudaMemcpyHostToDevice));

  // A2AMP communicates BF16, accumulates owner shards in FP32, and stores BF16 output.
  NCCL_CHECK(ncclAllReduceA2AMP(send, allReduceOutput, scratch, scratchBytes, count, ncclSum, comm, 0));
  NCCL_CHECK(ncclReduceScatterA2AMP(send, reduceScatterOutput, scratch, scratchBytes,
                                    recvcount, ncclSum, comm, 0));
  CUDA_CHECK(cudaDeviceSynchronize());

  const float expected = float(nranks * (nranks + 1) / 2);
  std::vector<__nv_bfloat16> allReduceHost(count), reduceScatterHost(recvcount);
  CUDA_CHECK(cudaMemcpy(allReduceHost.data(), allReduceOutput, count * sizeof(*allReduceOutput),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(reduceScatterHost.data(), reduceScatterOutput,
                        recvcount * sizeof(*reduceScatterOutput), cudaMemcpyDeviceToHost));
  int correct = 1;
  for (size_t i = 0; i < allReduceHost.size(); i++) {
    const float actual = __bfloat162float(allReduceHost[i]);
    if (actual != expected) {
      std::fprintf(stderr, "rank %d AllReduce[%zu]: got %g, expected %g\n", rank, i, actual, expected);
      correct = 0;
      break;
    }
  }
  for (size_t i = 0; i < reduceScatterHost.size(); i++) {
    const float actual = __bfloat162float(reduceScatterHost[i]);
    if (actual != expected) {
      std::fprintf(stderr, "rank %d ReduceScatter[%zu]: got %g, expected %g\n", rank, i, actual, expected);
      correct = 0;
      break;
    }
  }
  int allCorrect;
  MPI_CHECK(MPI_Allreduce(&correct, &allCorrect, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
  if (rank == 0)
    std::printf("A2AMP AllReduce and ReduceScatter: %s\n", allCorrect ? "PASS" : "FAIL");

  CUDA_CHECK(cudaFree(send));
  CUDA_CHECK(cudaFree(allReduceOutput));
  CUDA_CHECK(cudaFree(reduceScatterOutput));
  CUDA_CHECK(cudaFree(scratch));
  NCCL_CHECK(ncclCommDestroy(comm));
  MPI_CHECK(MPI_Finalize());
  return allCorrect ? EXIT_SUCCESS : EXIT_FAILURE;
}
