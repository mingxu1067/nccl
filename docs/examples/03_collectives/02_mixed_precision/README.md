<!--
  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for license information.
-->

# Mixed-Precision Collectives

These examples use one MPI process per GPU and demonstrate the BF16-input,
FP32-accumulation APIs:

- `c/ring_mp.cu`: `ncclAllReduceRingMP` and `ncclReduceScatterRingMP`
- `c/a2a_mp.cu`: `ncclAllReduceA2AMP` and `ncclReduceScatterA2AMP`

Both programs initialize BF16 input, run AllReduce and ReduceScatter, and check
the BF16 output.

## Build

First build NCCL, then pass its build directory as `NCCL_HOME`:

```shell
make -C nccl -j16 src.build \
  BUILDDIR="$PWD/artifacts/nccl-build-mixed-precision" \
  NVCC_GENCODE='-gencode=arch=compute_100,code=sm_100'

make -C nccl/docs/examples/03_collectives/02_mixed_precision/c \
  NCCL_HOME="$PWD/artifacts/nccl-build-mixed-precision" \
  MPI_HOME=/usr/local/mpi
```

## Run

Run four MPI ranks on four GPUs:

```shell
make -C nccl/docs/examples/03_collectives/02_mixed_precision/c run \
  NCCL_HOME="$PWD/artifacts/nccl-build-mixed-precision" \
  MPI_HOME=/usr/local/mpi NP=4
```

Expected output:

```text
RingMP AllReduce and ReduceScatter: PASS
A2AMP AllReduce and ReduceScatter: PASS
```

`RingMP` does not currently use its scratch arguments, so the examples pass
`nullptr, 0`. `A2AMP` requires BF16 scratch space equal to the full input size.
