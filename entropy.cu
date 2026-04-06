#include "dna_features.cuh"

#include <cstdint>

namespace {

__device__ __forceinline__ uint32_t encode_base_entropy(unsigned char b) {
    const uint32_t is_c = static_cast<uint32_t>(b == 'C' || b == 'c');
    const uint32_t is_g = static_cast<uint32_t>(b == 'G' || b == 'g');
    const uint32_t is_t = static_cast<uint32_t>(b == 'T' || b == 't');
    return (is_c * 1u) + (is_g * 2u) + (is_t * 3u);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__global__ void entropy_window_kernel(const char* __restrict__ d_sequence,
                                      int n,
                                      int k,
                                      int window,
                                      int bins,
                                      float min_entropy,
                                      float* __restrict__ d_entropy,
                                      int* __restrict__ d_entropy_violations,
                                      int* __restrict__ d_global_frequency) {
    extern __shared__ int s_hist[];
    __shared__ float s_warp_sums[32];

    const int tid = threadIdx.x;
    const int window_idx = blockIdx.x;
    const int n_windows = n - window + 1;

    if (window_idx >= n_windows) {
        return;
    }

    for (int b = tid; b < bins; b += blockDim.x) {
        s_hist[b] = 0;
    }
    __syncthreads();

    const int kmers_in_window = window - k + 1;

    // Build a per-window shared histogram to reduce global atomics pressure.
    for (int local_pos = tid; local_pos < kmers_in_window; local_pos += blockDim.x) {
        const int start = window_idx + local_pos;
        uint32_t hash = 0;
#pragma unroll 20
        for (int j = 0; j < 20; ++j) {
            if (j < k) {
                hash = (hash << 2) |
                       encode_base_entropy(static_cast<unsigned char>(d_sequence[start + j]));
            }
        }
        atomicAdd(&s_hist[hash], 1);
    }
    __syncthreads();

    // Merge each block histogram to global frequencies; this can be scanned later for CDF-like uses.
    for (int b = tid; b < bins; b += blockDim.x) {
        atomicAdd(&d_global_frequency[b], s_hist[b]);
    }
    __syncthreads();

    float local_entropy = 0.0f;
    for (int b = tid; b < bins; b += blockDim.x) {
        const int freq = s_hist[b];
        if (freq > 0) {
            const float p = static_cast<float>(freq) / static_cast<float>(kmers_in_window);
            local_entropy += -p * log2f(p);
        }
    }

    local_entropy = warp_reduce_sum(local_entropy);

    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    if (lane == 0) {
        s_warp_sums[warp_id] = local_entropy;
    }
    __syncthreads();

    float block_entropy = 0.0f;
    if (warp_id == 0) {
        const int num_warps = (blockDim.x + 31) >> 5;
        block_entropy = (lane < num_warps) ? s_warp_sums[lane] : 0.0f;
        block_entropy = warp_reduce_sum(block_entropy);

        if (lane == 0) {
            d_entropy[window_idx] = block_entropy;
            d_entropy_violations[window_idx] = static_cast<int>(block_entropy < min_entropy);
        }
    }
}

__global__ void zero_entropy_tail_kernel(float* __restrict__ d_entropy,
                                         int* __restrict__ d_entropy_violations,
                                         int n,
                                         int valid_windows) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x + valid_windows;
    if (idx < n) {
        d_entropy[idx] = 0.0f;
        d_entropy_violations[idx] = 0;
    }
}

}  // namespace

void run_entropy_feature(const char* d_sequence,
                         int n,
                         int k,
                         float min_entropy,
                         float* d_entropy,
                         int* d_entropy_violations,
                         cudaStream_t stream) {
    if (n <= 0) {
        return;
    }

    const int bins = 1 << (2 * k);
    const int n_windows = n - WINDOW_SIZE + 1;

    if (n_windows <= 0) {
        CUDA_CHECK(cudaMemsetAsync(d_entropy, 0, static_cast<size_t>(n) * sizeof(float), stream));
        CUDA_CHECK(
            cudaMemsetAsync(d_entropy_violations, 0, static_cast<size_t>(n) * sizeof(int), stream));
        return;
    }

    int* d_global_frequency = nullptr;
    int* d_global_frequency_scan = nullptr;
    CUDA_CHECK(cudaMalloc(&d_global_frequency, static_cast<size_t>(bins) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_global_frequency_scan, static_cast<size_t>(bins) * sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(d_global_frequency, 0, static_cast<size_t>(bins) * sizeof(int), stream));

    const int threads = 128;
    const int blocks = n_windows;
    const size_t shmem_bytes = static_cast<size_t>(bins) * sizeof(int);

    entropy_window_kernel<<<blocks, threads, shmem_bytes, stream>>>(d_sequence,
                                                                     n,
                                                                     k,
                                                                     WINDOW_SIZE,
                                                                     bins,
                                                                     min_entropy,
                                                                     d_entropy,
                                                                     d_entropy_violations,
                                                                     d_global_frequency);
    CUDA_CHECK(cudaGetLastError());

    // Prefix scan over merged global histogram enables fast cumulative distribution queries.
    launch_exclusive_scan_int(d_global_frequency, d_global_frequency_scan, bins, stream);

    const int tail_count = n - n_windows;
    if (tail_count > 0) {
        const int tail_threads = 256;
        const int tail_blocks = (tail_count + tail_threads - 1) / tail_threads;
        zero_entropy_tail_kernel<<<tail_blocks, tail_threads, 0, stream>>>(
            d_entropy, d_entropy_violations, n, n_windows);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(d_global_frequency));
    CUDA_CHECK(cudaFree(d_global_frequency_scan));
}
