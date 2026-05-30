#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

__global__ void gemm_kernel(int M, int N, int K, int layoutA, int layoutB, float* A, float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            float valA = (layoutA == 0) ? A[row * K + k] : A[k * M + row];
            float valB = (layoutB == 0) ? B[k * N + col] : B[col * K + k];
            sum += valA * valB;
        }
        C[row * N + col] = sum;
    }
}

//Function to read metadata and data from file
float* read_file(string filename, int &L, int &R, int &C_dim) {
    ifstream myFile(filename);
    myFile >> L >> R >> C_dim;
    float* data = new float[R * C_dim];
    for (int i = 0; i < R * C_dim; ++i) {
        myFile >> data[i];
    }
    myFile.close();
    return data;
}

void run_explicit_version(string fileA, string fileB) {
    int L_A, M, K, L_B, K_check, N;
    float *h_A = read_file(fileA, L_A, M, K);
    float *h_B = read_file(fileB, L_B, K_check, N);
    float *h_C = new float[M * N];
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    auto start = chrono::high_resolution_clock::now();

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (M + 15) / 16);
    gemm_kernel<<<gridSize, blockSize>>>(M, N, K, L_A, L_B, d_A, d_B, d_C);
    
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    cout << "Explicit memory allocation time: " << elapsed.count() << " ms; First element: " << h_C[0] << endl;

    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

void run_managed_version(string fileA, string fileB) {
    int L_A, M, K, L_B, K_check, N;

    ifstream fA(fileA), fB(fileB);
    fA >> L_A >> M >> K;
    fB >> L_B >> K_check >> N;

    float *A, *B, *C;
    cudaMallocManaged(&A, M * K * sizeof(float));
    cudaMallocManaged(&B, K * N * sizeof(float));
    cudaMallocManaged(&C, M * N * sizeof(float));

    for (int i = 0; i < M * K; ++i) fA >> A[i];
    for (int i = 0; i < K * N; ++i) fB >> B[i];

    auto start = chrono::high_resolution_clock::now();

    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (M + 15) / 16);
    gemm_kernel<<<gridSize, blockSize>>>(M, N, K, L_A, L_B, A, B, C);
    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    cout << "Unified memory allocation time: " << elapsed.count() << " ms; First element: " << C[0] << endl;

    cudaFree(A); cudaFree(B); cudaFree(C);
}

int main() {
    string fileA = "sample_0/A_64x64_T.txt";
    string fileB = "sample_0/B_64x64_T.txt";

    run_explicit_version(fileA, fileB);
    run_managed_version(fileA, fileB);

    return 0;
}