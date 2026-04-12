#include "dna_features.cuh"

#include <cstdint>

namespace {

__device__ __forceinline__ int is_gc_base(unsigned char b) {
    return static_cast<int>(b == 'G' || b == 'C' || b == 'g' || b == 'c');
}

__global__ void map_gc_kernel(const char* __restrict__ d_sequence,
                              int* __restrict__ d_gc_binary,
                              int n) {
    // Each thread reads 16 contiguous bytes via uint4, minimizing global transactions.
    const int vec_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int base = vec_idx * 16;

    if (base + 15 < n) {
        const uint4 packed = reinterpret_cast<const uint4*>(d_sequence)[vec_idx];
        const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&packed);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            d_gc_binary[base + j] = is_gc_base(bytes[j]);
        }
        return;
    }

    // Tail handling keeps accesses contiguous for the final partial vector.
    for (int idx = base; idx < n && idx < base + 16; ++idx) {
        d_gc_binary[idx] = is_gc_base(static_cast<unsigned char>(d_sequence[idx]));
    }
}

__global__ void prepare_scan_input_kernel(const int* __restrict__ d_gc_binary,
                                          int* __restrict__ d_scan_in,
                                          int n) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < n) {
        d_scan_in[idx] = d_gc_binary[idx];
    }
}

__global__ void gc_query_kernel(const int* __restrict__ d_gc_prefix,
                                int* __restrict__ d_gc_violations,
                                int n,
                                int window) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }

    // Keep out-of-range windows zeroed to simplify downstream aggregation logic.
    const int in_bounds = static_cast<int>(idx + window <= n);
    int violation = 0;

    if (in_bounds) {
        const int gc_count = d_gc_prefix[idx + window] - d_gc_prefix[idx];
        const float ratio = static_cast<float>(gc_count) / static_cast<float>(window);
        violation = static_cast<int>(ratio < GC_MIN || ratio > GC_MAX);
    }

    d_gc_violations[idx] = violation;
}

}  // namespace

void run_gc_content_feature(const char* d_sequence,
                            int n,
                            int window,
                            int* d_gc_violations,
                            int* d_gc_prefix,
                            cudaStream_t stream) {
    if (n <= 0) {
        return;
    }

    int* d_gc_binary = nullptr;
    int* d_scan_in = nullptr;

    CUDA_CHECK(cudaMalloc(&d_gc_binary, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scan_in, static_cast<size_t>(n + 1) * sizeof(int)));

    const int map_threads = 256;
    const int map_blocks = ((n + 15) / 16 + map_threads - 1) / map_threads;
    map_gc_kernel<<<map_blocks, map_threads, 0, stream>>>(d_sequence, d_gc_binary, n);
    CUDA_CHECK(cudaGetLastError());

    const int prep_threads = 256;
    const int prep_blocks = (n + prep_threads - 1) / prep_threads;
    prepare_scan_input_kernel<<<prep_blocks, prep_threads, 0, stream>>>(d_gc_binary, d_scan_in, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemsetAsync(d_scan_in + n, 0, sizeof(int), stream));

    // Exclusive scan over n+1 elements enables O(1) range GC queries at boundaries.
    launch_exclusive_scan_int(d_scan_in, d_gc_prefix, n + 1, stream);

    const int query_threads = 256;
    const int query_blocks = (n + query_threads - 1) / query_threads;
    gc_query_kernel<<<query_blocks, query_threads, 0, stream>>>(
        d_gc_prefix, d_gc_violations, n, window);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_gc_binary));
    CUDA_CHECK(cudaFree(d_scan_in));
}
