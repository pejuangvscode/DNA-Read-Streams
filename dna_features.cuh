#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr int WINDOW_SIZE = 50;
constexpr int MAX_HOMOPOLYMER = 3;
constexpr float GC_MIN = 0.45f;
constexpr float GC_MAX = 0.55f;
constexpr int NUM_BANKS = 32;
constexpr int NUM_BANK_BITS = 5;
constexpr int PIPELINE_CHUNK_SIZE = 1 << 20;  // 1M bases.

#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANK_BITS)

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err = (call);                                                       \
        if (err != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                         cudaGetErrorString(err));                                       \
            std::exit(1);                                                               \
        }                                                                                \
    } while (0)

struct FeatureResult {
    int* gc_violations;
    int* homopolymer_violations;
    int* motif_collisions;
    float* entropy_vector;
    int sequence_length;
};

struct PipelineTimings {
    float h2d_ms;
    float d2h_ms;
    float gc_ms;
    float homopolymer_ms;
    float motif_ms;
    float entropy_ms;
    float total_ms;
};

// Prefix scan wrappers.
void launch_exclusive_scan_int(const int* d_in, int* d_out, int n, cudaStream_t stream = 0);
void launch_exclusive_scan_float(const float* d_in, float* d_out, int n, cudaStream_t stream = 0);
void launch_segmented_exclusive_scan_int(const int* d_values,
                                         const int* d_head_flags,
                                         int* d_out,
                                         int n,
                                         cudaStream_t stream = 0);

// Feature kernels.
void run_gc_content_feature(const char* d_sequence,
                            int n,
                            int window,
                            int* d_gc_violations,
                            int* d_gc_prefix,
                            cudaStream_t stream = 0);

void run_homopolymer_feature(const char* d_sequence,
                             int n,
                             int* d_homopolymer_violations,
                             int* d_run_length,
                             cudaStream_t stream = 0);

void set_forbidden_motifs(const uint64_t* h_hashes, int count, int k);
void run_motif_feature(const char* d_sequence,
                       int n,
                       int k,
                       int* d_motif_collisions,
                       int* d_motif_prefix,
                       cudaStream_t stream = 0);

void run_entropy_feature(const char* d_sequence,
                         int n,
                         int k,
                         float min_entropy,
                         float* d_entropy,
                         int* d_entropy_violations,
                         cudaStream_t stream = 0);

// Orchestration API.
void run_feature_pipeline_async(const char* h_sequence,
                                int length,
                                int motif_k,
                                int entropy_k,
                                float min_entropy,
                                const std::vector<uint64_t>& forbidden_hashes,
                                FeatureResult* h_result,
                                PipelineTimings* timings);
