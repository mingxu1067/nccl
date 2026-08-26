#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for license information.
#

NCCL_HOME ?= ../../../../../../build
MPI_HOME ?= /usr/local/mpi
NVCC ?= nvcc
MPIRUN ?= $(MPI_HOME)/bin/mpirun
NP ?= 4
PREFIX ?= /usr/local

CPPFLAGS = -I$(NCCL_HOME)/include -I$(MPI_HOME)/include -I$(MPI_HOME)/include/openmpi
LDFLAGS = -L$(NCCL_HOME)/lib -L$(MPI_HOME)/lib -lnccl -lmpi
TARGETS = ring_mp a2a_mp

all: $(TARGETS)

%: %.cu
	$(NVCC) -std=c++17 $(CPPFLAGS) $< $(LDFLAGS) -o $@

run: all
	LD_LIBRARY_PATH=$(NCCL_HOME)/lib:$(MPI_HOME)/lib:$${LD_LIBRARY_PATH} \
	  $(MPIRUN) --allow-run-as-root --oversubscribe -np $(NP) ./ring_mp
	LD_LIBRARY_PATH=$(NCCL_HOME)/lib:$(MPI_HOME)/lib:$${LD_LIBRARY_PATH} \
	  $(MPIRUN) --allow-run-as-root --oversubscribe -np $(NP) ./a2a_mp

clean:
	rm -f $(TARGETS)

test: run

install: all
	mkdir -p $(PREFIX)/bin
	cp $(TARGETS) $(PREFIX)/bin/

.PHONY: all run test install clean
