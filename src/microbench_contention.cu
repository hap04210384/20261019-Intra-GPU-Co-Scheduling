// E15: contention microbenchmark — quantify mutual slowdown when a CUDA-core
// (SIMT) mask-building kernel and a tensor-core (WMMA) kernel run concurrently
// on two streams, vs each running alone. This bounds the Mode-E overlap loss.
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <chrono>
using namespace nvcuda;

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

// ---- SIMT-side kernel: bitmap mask build (memory-bound AND+shuffle) ----
__global__ void maskKernel(const unsigned long long* __restrict__ A,
                           const unsigned long long* __restrict__ B,
                           unsigned long long* __restrict__ C, long words) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (; i < words; i += stride) C[i] = A[i] & B[i];
}

// ---- tensor-side kernel: HMMA sustained on register fragments ----
__global__ void mmaKernel(float* out, int iters) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c, d;
    wmma::fill_fragment(a, __float2half(1.0f));
    wmma::fill_fragment(b, __float2half(1.0f));
    wmma::fill_fragment(c, 0.0f);
    wmma::fill_fragment(d, 0.0f);
    for (int it = 0; it < iters; it++) {
        wmma::mma_sync(c, a, b, c);
        wmma::mma_sync(d, a, b, d);
        wmma::mma_sync(c, a, b, d);
        wmma::mma_sync(d, a, b, c);
    }
    if (threadIdx.x == 1024) out[0] = c.x[0] + d.x[0];
}

static float timeIt(void (*launch)(int, cudaStream_t), int arg, cudaStream_t s) {
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    launch(arg, s); cudaStreamSynchronize(s);            // warm
    cudaEventRecord(e0, s);
    launch(arg, s);
    cudaEventRecord(e1, s);
    cudaStreamSynchronize(s);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}

static long g_words; static unsigned long long *g_A, *g_B, *g_C; static float* g_out; static int g_mmaIters, g_mmaBlocks;
static void launchMask(int, cudaStream_t s) { maskKernel<<<38*8, 256, 0, s>>>(g_A, g_B, g_C, g_words); }
static void launchMma(int, cudaStream_t s)   { mmaKernel<<<g_mmaBlocks, 256, 0, s>>>(g_out, g_mmaIters); }

int main() {
    CK(cudaSetDevice(0));
    g_words = 96LL * 1024 * 1024 / 8;              // 96 MB per array
    CK(cudaMalloc(&g_A, g_words * 8)); CK(cudaMalloc(&g_B, g_words * 8)); CK(cudaMalloc(&g_C, g_words * 8));
    CK(cudaMalloc(&g_out, 4));
    CK(cudaMemset(g_A, 0x5A, g_words * 8)); CK(cudaMemset(g_B, 0xC3, g_words * 8));
    g_mmaIters = 20000; g_mmaBlocks = 38 * 4;      // 152 blocks of 8 warps: sustained HMMA

    cudaStream_t sMask, sMma;
    CK(cudaStreamCreate(&sMask)); CK(cudaStreamCreate(&sMma));

    std::vector<float> tMa, tMb, tCa, tCb;
    for (int r = 0; r < 15; r++) {
        tMa.push_back(timeIt(launchMask, 0, sMask));
        tMb.push_back(timeIt(launchMma, 0, sMma));
        // concurrent: launch both, time each with events on its stream
        cudaEvent_t e0a, e1a, e0b, e1b;
        cudaEventCreate(&e0a); cudaEventCreate(&e1a); cudaEventCreate(&e0b); cudaEventCreate(&e1b);
        cudaEventRecord(e0a, sMask);
        launchMask(0, sMask);
        cudaEventRecord(e1a, sMask);
        cudaEventRecord(e0b, sMma);
        launchMma(0, sMma);
        cudaEventRecord(e1b, sMma);
        cudaStreamSynchronize(sMask); cudaStreamSynchronize(sMma);
        float a, b; cudaEventElapsedTime(&a, e0a, e1a); cudaEventElapsedTime(&b, e0b, e1b);
        tCa.push_back(a); tCb.push_back(b);
        cudaEventDestroy(e0a); cudaEventDestroy(e1a); cudaEventDestroy(e0b); cudaEventDestroy(e1b);
    }
    auto med = [](std::vector<float>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    float ma = med(tMa), mb = med(tMb), ca = med(tCa), cb = med(tCb);
    printf("E15 contention microbench (RTX 3060 Ti, medians of 15)\n");
    printf("mask alone : %.3f ms\n", ma);
    printf("mma  alone : %.3f ms\n", mb);
    printf("mask conc. : %.3f ms  (stretch x%.3f)\n", ca, ca / ma);
    printf("mma  conc. : %.3f ms  (stretch x%.3f)\n", cb, cb / mb);
    printf("sum alone  : %.3f ms;  conc max: %.3f ms  (overlap saving %.1f%%)\n",
           ma + mb, std::max(ca, cb), 100.0 * (1.0 - std::max(ca, cb) / (ma + mb)));
    return 0;
}
