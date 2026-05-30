#include <cuda.h>
#include <cuda_fp16.h>
#include "mma_intrinsics.cuh"

#include <cuda_fp16.h>
#include "mma_intrinsics.cuh"

#include "mma_intrinsics.cuh"
#include <cuda_fp16.h>

__global__ void tensorCoreGemmKernel(half *A, half *B, float *C, int M, int N, int K) {
    int row_C = blockIdx.x * 16;
    int col_C = blockIdx.y * 8;

    __shared__ half s_A[256];   
    __shared__ half s_B[128];   
    __shared__ float s_C[128];  

    for (int i = threadIdx.x; i < 128; i += 32) {
        s_C[i] = 0.0f;
    }
    __syncthreads(); 

    for (int k = 0; k < K; k += 16) {
        
        for (int i = threadIdx.x; i < 256; i += 32) {
            int r = i / 16;
            int c = i % 16;
            s_A[i] = A[(row_C + r) * K + (k + c)];
        }

        for (int i = threadIdx.x; i < 128; i += 32) {
            int c = i / 16; 
            int r = i % 16; 

            s_B[i] = B[(col_C + c) * K + (k + r)]; 
        }

        __syncwarp(); 

        mma_m16n8k16_f16_f16_smem_row_col(s_A, s_B, s_C);

        __syncwarp(); 
    }

    for (int i = threadIdx.x; i < 128; i += 32) {
        int r = i / 8;
        int c = i % 8;
        C[(row_C + r) * N + (col_C + c)] = s_C[i];
    }
}

void launchStudentKernel(int M, int N, int K, half* A, half* B, float* C) {
    
    dim3 blockDim(32); 
    
    dim3 gridDim(M / 16, N / 8); 

    tensorCoreGemmKernel<<<gridDim, blockDim>>>(A, B, C, M, N, K);
}