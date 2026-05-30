#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>
#include "mma_intrinsics.cuh"

__global__ void GEMM(half* A, half* B, float* C, int M, int N, int K) {

    int tile_row = blockIdx.y * 64;
    int tile_col = blockIdx.x * 64;
    int tid = threadIdx.x;

    extern __shared__ __align__(128) int8_t smem[];

    half* smem_A = (half*)smem;
    half* smem_B = smem_A + 8192;
    float* smem_C = (float*)(smem_B + 8192);

    for (int i = tid; i < 4096; i += blockDim.x)
        smem_C[i] = 0.0f;
    __syncthreads();

    int row_in_tile = tid / 2;
    int col_in_tile = (tid % 2) * 32;
    int smem_idx = row_in_tile * 64 + col_in_tile;

    int num_batches = K / 64;
    auto pipeline = cuda::make_pipeline();

    // PRIME
    pipeline.producer_acquire();
    cuda::memcpy_async(smem_A + smem_idx, &A[(tile_row + row_in_tile) * K + col_in_tile], sizeof(half) * 32, pipeline);
    cuda::memcpy_async(smem_B + smem_idx, &B[(tile_col + row_in_tile) * K + col_in_tile], sizeof(half) * 32, pipeline);
    pipeline.producer_commit();

    // LOOP
    for (int batch = 1; batch < num_batches; batch++) {
        int copy_offset    = (batch % 2) * 4096;
        int compute_offset = ((batch - 1) % 2) * 4096;

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

    for (int i = tid; i < 4096; i += blockDim.x) {
        int r = i / 64;
        int c = i % 64;
        if (tile_row + r < M && tile_col + c < N)
            C[(tile_row + r) * N + (tile_col + c)] = smem_C[i];
    }
}

static void launch_gemm(half* d_A, half* d_B, float* d_C, int M, int N, int K) {
    dim3 blockDim(128);
    dim3 gridDim(N / 64, M / 64);
    int smemBytes = 65536;
    cudaFuncSetAttribute(GEMM, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
    GEMM<<<gridDim, blockDim, smemBytes>>>(d_A, d_B, d_C, M, N, K);
}

static void init_matrix_half(half* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        mat[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
}

// Read a matrix file: line1=rows, line2=cols, then rows*cols float values.
// Returns false on failure.
static bool read_matrix_file(const char* path, int& rows, int& cols, float*& data) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open " << path << "\n";
        return false;
    }
    fin >> rows >> cols;
    data = (float*)malloc((size_t)rows * cols * sizeof(float));
    for (int i = 0; i < rows * cols; i++) {
        if (!(fin >> data[i])) {
            std::cerr << "Error: unexpected EOF in " << path << "\n";
            free(data); data = nullptr;
            return false;
        }
    }
    return true;
}

// Write a matrix file in the same format.
static void write_matrix_file(const char* path, int rows, int cols, const float* data) {
    std::ofstream fout(path);
    fout << rows << "\n" << cols << "\n";
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (c) fout << ' ';
            fout << data[r * cols + c];
        }
        fout << '\n';
    }
    std::cout << "Wrote " << path << "\n";
}

// Run timed GEMM with random inputs.
static void run_timing(int M, int N, int K) {
    size_t sz_A = (size_t)M * K * sizeof(half);
    size_t sz_B = (size_t)K * N * sizeof(half);
    size_t sz_C = (size_t)M * N * sizeof(float);

    half*  h_A = (half*)malloc(sz_A);
    half*  h_B = (half*)malloc(sz_B);
    float* h_C = (float*)malloc(sz_C);
    init_matrix_half(h_A, M, K);
    init_matrix_half(h_B, K, N);

    half *d_A, *d_B; float *d_C;
    cudaMalloc(&d_A, sz_A);
    cudaMalloc(&d_B, sz_B);
    cudaMalloc(&d_C, sz_C);
    cudaMemcpy(d_A, h_A, sz_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sz_B, cudaMemcpyHostToDevice);

    // Warm-up
    launch_gemm(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    launch_gemm(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "GEMM Dimensions: M=" << M << " N=" << N << " K=" << K << "\n";
    std::cout << "Pipelined Tensor Core Kernel Execution Time: " << ms << " ms\n";

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

// Read A.txt and B.txt, run GEMM, write C_.txt.
//
// File format (same for A, B, C):
//   Line 1: rows
//   Line 2: cols
//   Then rows*cols space-separated float values, row by row.
//
// A is MxK row-major    → rows=M, cols=K, flat order is row-major.
// B is KxN column-major → rows=K, cols=N, but each file-row is a column of B,
//                          so reading flat gives the column-major memory layout directly.
// C is MxN row-major    → rows=M, cols=N.
static void run_from_ab_files(const char* a_path, const char* b_path) {
    float *fa = nullptr, *fb = nullptr;
    int aR, aC, bR, bC;

    if (!read_matrix_file(a_path, aR, aC, fa)) return;
    if (!read_matrix_file(b_path, bR, bC, fb)) { free(fa); return; }

    // aR=M, aC=K, bR=K, bC=N
    int M = aR, K = aC, N = bC;
    if (bR != K) {
        std::cerr << "Error: A columns (" << K << ") != B rows (" << bR << ")\n";
        free(fa); free(fb); return;
    }

    std::cout << "M=" << M << " K=" << K << " N=" << N << "\n";

    // Convert A (float, row-major) → half row-major
    half* h_A = (half*)malloc((size_t)M * K * sizeof(half));
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(fa[i]);
    free(fa);

    // B file is stored row-major (file[k][n] = B[k][n]).
    // The kernel needs B in column-major (h_B[n*K + k] = B[k][n]), so transpose.
    half* h_B = (half*)malloc((size_t)K * N * sizeof(half));
    for (int k = 0; k < K; k++)
        for (int n = 0; n < N; n++)
            h_B[(size_t)n * K + k] = __float2half(fb[(size_t)k * N + n]);
    free(fb);

    float* h_C = (float*)malloc((size_t)M * N * sizeof(float));

    half *d_A, *d_B; float *d_C;
    cudaMalloc(&d_A, (size_t)M * K * sizeof(half));
    cudaMalloc(&d_B, (size_t)K * N * sizeof(half));
    cudaMalloc(&d_C, (size_t)M * N * sizeof(float));
    cudaMemcpy(d_A, h_A, (size_t)M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, (size_t)K * N * sizeof(half), cudaMemcpyHostToDevice);

    launch_gemm(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, (size_t)M * N * sizeof(float), cudaMemcpyDeviceToHost);

    write_matrix_file("C_.txt", M, N, h_C);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}

int main(int argc, char** argv) {
    if (argc == 3) {
        run_from_ab_files(argv[1], argv[2]);
    } else if (argc == 4) {
        int M = std::stoi(argv[1]);
        int N = std::stoi(argv[2]);
        int K = std::stoi(argv[3]);
        run_timing(M, N, K);
    } else {
        std::cerr << "Usage:\n"
                  << "  " << argv[0] << " <M> <N> <K>         -- random timing run\n"
                  << "  " << argv[0] << " <A.txt> <B.txt>     -- compute C from files, write C_.txt\n";
        return 1;
    }
    return 0;
}
