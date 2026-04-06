#include "dna_features.cuh"

#include <algorithm>
#include <cstdint>

namespace {

constexpr int kMaxForbiddenMotifs = 2048;

__constant__ uint64_t c_forbidden_hashes[kMaxForbiddenMotifs];
__constant__ int c_num_forbidden;
__constant__ int c_motif_k;

__device__ __forceinline__ uint32_t encode_base_2bit(unsigned char b) {
    // A=00, C=01, G=10, T=11
    const uint32_t is_c = static_cast<uint32_t>(b == 'C' || b == 'c');
    const uint32_t is_g = static_cast<uint32_t>(b == 'G' || b == 'g');
    const uint32_t is_t = static_cast<uint32_t>(b == 'T' || b == 't');
    return (is_c * 1u) + (is_g * 2u) + (is_t * 3u);
}

__global__ void rolling_hash_match_kernel(const char* __restrict__ d_sequence,
                                          int* __restrict__ d_matches,
                                          int n,
                                          int k) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx >= n) {
        return;
    }

    int match = 0;
    if (idx + k <= n) {
        uint64_t hash64 = 0;

        if (k <= 16) {
            // k<=16 can be kept in 32 bits; cast once to 64 for unified compare path.
            uint32_t hash32 = 0;
#pragma unroll 16
            for (int j = 0; j < 16; ++j) {
                if (j < k) {
                    hash32 = (hash32 << 2) | encode_base_2bit(static_cast<unsigned char>(d_sequence[idx + j]));
                }
            }
            hash64 = static_cast<uint64_t>(hash32);
        } else {
#pragma unroll 20
            for (int j = 0; j < 20; ++j) {
                if (j < k) {
                    hash64 = (hash64 << 2) | encode_base_2bit(static_cast<unsigned char>(d_sequence[idx + j]));
                }
            }
        }

        // Constant memory broadcast is efficient when all threads inspect motif table entries.
        for (int m = 0; m < c_num_forbidden; ++m) {
            if (hash64 == c_forbidden_hashes[m]) {
                match = 1;
                break;
            }
        }
    }

    d_matches[idx] = match;
}

__global__ void prepare_scan_input_kernel(const int* __restrict__ d_matches,
                                          int* __restrict__ d_scan_in,
                                          int n) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < n) {
        d_scan_in[idx] = d_matches[idx];
    }
}

}  // namespace

void set_forbidden_motifs(const uint64_t* h_hashes, int count, int k) {
    const int safe_count = std::max(0, std::min(count, kMaxForbiddenMotifs));
    CUDA_CHECK(cudaMemcpyToSymbol(c_num_forbidden, &safe_count, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_motif_k, &k, sizeof(int)));
    if (safe_count > 0) {
        CUDA_CHECK(cudaMemcpyToSymbol(
            c_forbidden_hashes, h_hashes, static_cast<size_t>(safe_count) * sizeof(uint64_t)));
    }
}

void run_motif_feature(const char* d_sequence,
                       int n,
                       int k,
                       int* d_motif_collisions,
                       int* d_motif_prefix,
                       cudaStream_t stream) {
    if (n <= 0) {
        return;
    }

    int* d_scan_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scan_in, static_cast<size_t>(n + 1) * sizeof(int)));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    rolling_hash_match_kernel<<<blocks, threads, 0, stream>>>(d_sequence, d_motif_collisions, n, k);
    CUDA_CHECK(cudaGetLastError());

    prepare_scan_input_kernel<<<blocks, threads, 0, stream>>>(d_motif_collisions, d_scan_in, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemsetAsync(d_scan_in + n, 0, sizeof(int), stream));

    launch_exclusive_scan_int(d_scan_in, d_motif_prefix, n + 1, stream);

    CUDA_CHECK(cudaFree(d_scan_in));
}
