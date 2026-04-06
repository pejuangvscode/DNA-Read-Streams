#include "dna_features.cuh"

namespace {

__global__ void homopolymer_compare_kernel(const char* __restrict__ d_sequence,
                                           int* __restrict__ d_h,
                                           int n) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }

    // H[0] = 0, and for idx>0 this is 1 when current base extends the run.
    const int same_as_prev = static_cast<int>((idx > 0) && (d_sequence[idx] == d_sequence[idx - 1]));
    d_h[idx] = same_as_prev;
}

__global__ void build_segment_heads_kernel(const int* __restrict__ d_h,
                                           int* __restrict__ d_head_flags,
                                           int n) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < n) {
        // Branchless reset condition: 1 when H[i] == 0, else 0.
        d_head_flags[idx] = 1 - d_h[idx];
    }
}

__global__ void build_run_length_kernel(const int* __restrict__ d_h,
                                        const int* __restrict__ d_segmented_exclusive,
                                        int* __restrict__ d_run_length,
                                        int n) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < n) {
        // Run length increases only when H[i] == 1; starts remain at 0.
        d_run_length[idx] = d_h[idx] * (d_segmented_exclusive[idx] + 1);
    }
}

__global__ void homopolymer_threshold_kernel(const int* __restrict__ d_run_length,
                                             int* __restrict__ d_violations,
                                             int n) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < n) {
        d_violations[idx] = static_cast<int>(d_run_length[idx] > MAX_HOMOPOLYMER);
    }
}

}  // namespace

void run_homopolymer_feature(const char* d_sequence,
                             int n,
                             int* d_homopolymer_violations,
                             int* d_run_length,
                             cudaStream_t stream) {
    if (n <= 0) {
        return;
    }

    int* d_h = nullptr;
    int* d_head_flags = nullptr;
    int* d_segmented_exclusive = nullptr;
    int* d_run_length_internal = d_run_length;

    CUDA_CHECK(cudaMalloc(&d_h, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_head_flags, static_cast<size_t>(n) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_segmented_exclusive, static_cast<size_t>(n) * sizeof(int)));

    if (d_run_length_internal == nullptr) {
        CUDA_CHECK(cudaMalloc(&d_run_length_internal, static_cast<size_t>(n) * sizeof(int)));
    }

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    homopolymer_compare_kernel<<<blocks, threads, 0, stream>>>(d_sequence, d_h, n);
    CUDA_CHECK(cudaGetLastError());

    build_segment_heads_kernel<<<blocks, threads, 0, stream>>>(d_h, d_head_flags, n);
    CUDA_CHECK(cudaGetLastError());

    launch_segmented_exclusive_scan_int(d_h, d_head_flags, d_segmented_exclusive, n, stream);

    build_run_length_kernel<<<blocks, threads, 0, stream>>>(
        d_h, d_segmented_exclusive, d_run_length_internal, n);
    CUDA_CHECK(cudaGetLastError());

    homopolymer_threshold_kernel<<<blocks, threads, 0, stream>>>(
        d_run_length_internal, d_homopolymer_violations, n);
    CUDA_CHECK(cudaGetLastError());

    if (d_run_length == nullptr) {
        CUDA_CHECK(cudaFree(d_run_length_internal));
    }

    CUDA_CHECK(cudaFree(d_h));
    CUDA_CHECK(cudaFree(d_head_flags));
    CUDA_CHECK(cudaFree(d_segmented_exclusive));
}
