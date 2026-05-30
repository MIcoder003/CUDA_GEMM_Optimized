#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include <cuda/barrier>
// Disables `cuda::barrier` initialization warning.
#pragma nv_diag_suppress static_var_with_dynamic_init

#include "mma_intrinsics.cuh"
// TODO: Implement this function...
// This is for grading. You will use this function with the testbench we provide.
// You can add more functions etc. here if you want.
// You only need to launch your kernel inside this function, everything else will be managed by the testbench.
// M, N, K are matrix dimensions
// A is row major, B is column major, C is row major
// A, B and C are pointers to the matrices

__global__ void GEMM(half* A, half* B, float* C, int M, int N, int K) {

    int tile_row = blockIdx.y * 64;
    int tile_col = blockIdx.x * 64;
    int tid = threadIdx.x;

    extern __shared__ __align__(128) int8_t smem[];
    
    half* smem_A = (half*)smem;
    half* smem_B = smem_A + 8192;
    float* smem_C = (float*)(smem_B + 8192);

    for (int i = tid; i < 4096; i += blockDim.x) {
        smem_C[i] = 0.0f;
    }
    __syncthreads(); 

    int row_in_tile = tid / 2;
    int col_in_tile = (tid % 2) * 32;
    int smem_idx = row_in_tile * 64 + col_in_tile;

    int num_batches = K / 64;
    auto pipeline = cuda::make_pipeline();

    // PRIME 
    pipeline.producer_acquire();
    // For prime, we use slot 0 (offset 0)
    cuda::memcpy_async(smem_A + smem_idx, &A[(tile_row + row_in_tile) * K + (0 + col_in_tile)], sizeof(half) * 32, pipeline);
    cuda::memcpy_async(smem_B + smem_idx, &B[(tile_col + row_in_tile) * K + (0 + col_in_tile)], sizeof(half) * 32, pipeline);
    pipeline.producer_commit();

    // LOOP
    for (int batch = 1; batch < num_batches; batch++) {
        int copy_slot = batch % 2;
        int compute_slot = (batch - 1) % 2;

        // Calculate the base offsets for this stage (0 or 4096)
        int copy_offset = copy_slot * 4096;
        int compute_offset = compute_slot * 4096;

        pipeline.producer_acquire();
        cuda::memcpy_async(smem_A + copy_offset + smem_idx, &A[(tile_row + row_in_tile) * K + (batch * 64 + col_in_tile)], sizeof(half) * 32, pipeline);
        cuda::memcpy_async(smem_B + copy_offset + smem_idx, &B[(tile_col + row_in_tile) * K + (batch * 64 + col_in_tile)], sizeof(half) * 32, pipeline);
        pipeline.producer_commit(); 

        pipeline.consumer_wait();
        __syncthreads();
        mma_m16n8k16_f16_f16_smem_row_col_64x64(smem_A + compute_offset, smem_B + compute_offset, smem_C);
        pipeline.consumer_release();
    }

    // DRAIN
    int final_compute_offset = ((num_batches - 1) % 2) * 4096;
    pipeline.consumer_wait();
    __syncthreads();
    mma_m16n8k16_f16_f16_smem_row_col_64x64(smem_A + final_compute_offset, smem_B + final_compute_offset, smem_C);
    pipeline.consumer_release();

    __syncthreads(); 

    // Write the fully computed smem_C tile back to the global C matrix
    for (int i = tid; i < 4096; i += blockDim.x) {
        int r = i / 64;
        int c = i % 64;
        
        if (tile_row + r < M && tile_col + c < N) {
            C[(tile_row + r) * N + (tile_col + c)] = smem_C[i];
        }
    }
}

void launchStudentKernel(int M, int N, int K, half* d_A, half* d_B, float* d_C) {

    dim3 blockDim(128);
    dim3 gridDim(N / 64, M / 64);
    
    int smemBytes = 65536;
    cudaFuncSetAttribute(GEMM, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
    
    GEMM<<<gridDim, blockDim, smemBytes>>>(d_A, d_B, d_C, M, N, K);
}