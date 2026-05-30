# CUDA_GEMM_Optimized

A progressive optimization study of General Matrix Multiplication (GEMM) on NVIDIA GPUs using CUDA C++. This project implements three increasingly sophisticated kernel strategies — from a baseline implementation to tensor core acceleration and software pipelining — demonstrating hands-on GPU architecture knowledge and high-performance computing techniques.

---

## Hardware

Developed and benchmarked on the **Hydra HPC Cluster** at NC State University's Electrical and Computer Engineering (ECE) department.

| Spec | Details |
|------|---------|
| GPU | NVIDIA Titan Xp (dual per server) |
| VRAM | 12GB GDDR5X per card |
| System RAM | 128GB per server |
| OS | Linux |
| Cluster | 3 servers total |

---

## Project Structure

```
CUDA_GEMM_Optimized/
├── 01_baseline_kernel/         # Naive GEMM kernel — tiled shared memory
│   ├── main.cu
│   ├── launchStudentKernel.cu
│   └── Makefile
│
├── 02_tensor_optimized/        # Tensor core acceleration using WMMA intrinsics
│   ├── main.cu
│   ├── launchStudentKernel.cu
│   ├── mma_intrinsics.cuh
│   └── Makefile
│
└── 03_software_pipelining/     # Async memory pipelining to hide latency
    ├── main.cu
    ├── launchStudentKernel.cu
    ├── mma_intrinsics.cuh
    ├── C_64x64.txt
    ├── pa4_testbench
    └── Makefile
```

---

## Optimization Stages

### 01 — Baseline Kernel
A tiled GEMM kernel using shared memory to reduce global memory traffic. Establishes the performance baseline and demonstrates core CUDA programming fundamentals including thread/block indexing and shared memory synchronization.

### 02 — Tensor Core Optimization
Leverages NVIDIA's Warp Matrix Multiply Accumulate (WMMA) API and MMA intrinsics to offload computation onto the Titan Xp's tensor cores. Achieves significantly higher throughput by exploiting hardware-level matrix acceleration.

### 03 — Software Pipelining
Introduces asynchronous memory operations to overlap data movement with computation. By staging data loads and hiding memory latency through double-buffering, this stage pushes utilization closer to peak theoretical throughput.

---

## Build & Run

Each stage has its own `Makefile`. To build and run any stage:

```bash
cd 01_baseline_kernel   # or 02_tensor_optimized / 03_software_pipelining
make
./main
```

> **Note:** Requires CUDA toolkit and an NVIDIA GPU with compute capability 7.0+ for tensor core stages.

---

## Key Concepts Demonstrated

- CUDA thread hierarchy — grids, blocks, warps
- Shared memory tiling and bank conflict avoidance
- Tensor core programming via WMMA / MMA intrinsics
- Software pipelining and async memory prefetching
- Makefile-based CUDA build systems

---

## Environment

- **CUDA Toolkit:** 10.x+
- **Compiler:** `nvcc`
- **Language:** CUDA C++

---

*Developed as part of advanced GPU architecture coursework at NC State University — ECE Department.*
