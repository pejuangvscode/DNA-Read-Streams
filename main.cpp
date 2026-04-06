#include "dna_features.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

uint32_t encode_base_cpu(char b) {
    switch (b) {
        case 'C':
        case 'c':
            return 1u;
        case 'G':
        case 'g':
            return 2u;
        case 'T':
        case 't':
            return 3u;
        default:
            return 0u;
    }
}

uint64_t hash_kmer_cpu(const char* seq, int start, int k) {
    uint64_t h = 0;
    for (int i = 0; i < k; ++i) {
        h = (h << 2) | encode_base_cpu(seq[start + i]);
    }
    return h;
}

void cpu_gc_reference_chunked(const char* seq, int n, int chunk_size, std::vector<int>& out) {
    out.assign(n, 0);

    for (int offset = 0; offset < n; offset += chunk_size) {
        const int len = std::min(chunk_size, n - offset);
        std::vector<int> prefix(len + 1, 0);
        for (int i = 0; i < len; ++i) {
            const char b = seq[offset + i];
            const int is_gc = (b == 'G' || b == 'C' || b == 'g' || b == 'c') ? 1 : 0;
            prefix[i + 1] = prefix[i] + is_gc;
        }

        for (int i = 0; i < len; ++i) {
            if (i + WINDOW_SIZE <= len) {
                const int gc_count = prefix[i + WINDOW_SIZE] - prefix[i];
                const float ratio = static_cast<float>(gc_count) / static_cast<float>(WINDOW_SIZE);
                out[offset + i] = static_cast<int>(ratio < GC_MIN || ratio > GC_MAX);
            }
        }
    }
}

void cpu_homopolymer_reference_chunked(const char* seq, int n, int chunk_size, std::vector<int>& out) {
    out.assign(n, 0);

    for (int offset = 0; offset < n; offset += chunk_size) {
        const int len = std::min(chunk_size, n - offset);
        std::vector<int> h(len, 0);
        std::vector<int> run(len, 0);

        for (int i = 1; i < len; ++i) {
            h[i] = static_cast<int>(seq[offset + i] == seq[offset + i - 1]);
        }

        int segmented = 0;
        for (int i = 0; i < len; ++i) {
            if (h[i] == 0) {
                segmented = 0;
                run[i] = 0;
            } else {
                run[i] = segmented + 1;
                segmented = run[i];
            }
            out[offset + i] = static_cast<int>(run[i] > MAX_HOMOPOLYMER);
        }
    }
}

void cpu_motif_reference_chunked(const char* seq,
                                 int n,
                                 int chunk_size,
                                 int k,
                                 const std::unordered_set<uint64_t>& forbidden,
                                 std::vector<int>& out) {
    out.assign(n, 0);

    for (int offset = 0; offset < n; offset += chunk_size) {
        const int len = std::min(chunk_size, n - offset);

        for (int i = 0; i < len; ++i) {
            if (i + k <= len) {
                const uint64_t h = hash_kmer_cpu(seq + offset, i, k);
                out[offset + i] = static_cast<int>(forbidden.find(h) != forbidden.end());
            }
        }
    }
}

void cpu_entropy_reference_chunked(const char* seq,
                                   int n,
                                   int chunk_size,
                                   int k,
                                   std::vector<float>& out) {
    out.assign(n, 0.0f);
    const int bins = 1 << (2 * k);

    for (int offset = 0; offset < n; offset += chunk_size) {
        const int len = std::min(chunk_size, n - offset);
        const int valid_windows = len - WINDOW_SIZE + 1;
        if (valid_windows <= 0) {
            continue;
        }

        std::vector<int> hist(bins, 0);
        for (int i = 0; i < valid_windows; ++i) {
            std::fill(hist.begin(), hist.end(), 0);
            const int kmers = WINDOW_SIZE - k + 1;
            for (int p = 0; p < kmers; ++p) {
                const uint64_t h = hash_kmer_cpu(seq + offset + i, p, k);
                hist[static_cast<int>(h)] += 1;
            }

            float entropy = 0.0f;
            for (int b = 0; b < bins; ++b) {
                if (hist[b] > 0) {
                    const float prob = static_cast<float>(hist[b]) / static_cast<float>(kmers);
                    entropy += -prob * std::log2(prob);
                }
            }
            out[offset + i] = entropy;
        }
    }
}

double bases_per_second(int bases, float ms) {
    if (ms <= 0.0f) {
        return 0.0;
    }
    return static_cast<double>(bases) / (static_cast<double>(ms) * 1.0e-3);
}

double gb_per_second(size_t bytes, float ms) {
    if (ms <= 0.0f) {
        return 0.0;
    }
    return (static_cast<double>(bytes) / 1.0e9) / (static_cast<double>(ms) * 1.0e-3);
}

}  // namespace

int main() {
    constexpr int kSequenceLength = 10'000'000;
    constexpr int kMotifK = 8;
    constexpr int kEntropyK = 3;
    constexpr float kMinEntropy = 1.60f;

    char* h_sequence = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_sequence, static_cast<size_t>(kSequenceLength) * sizeof(char)));

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 3);
    const char alphabet[4] = {'A', 'C', 'G', 'T'};
    for (int i = 0; i < kSequenceLength; ++i) {
        h_sequence[i] = alphabet[dist(rng)];
    }

    std::vector<std::string> forbidden_motifs = {"AAAA", "CCCC", "GGGG", "TTTT", "ACGTACGT"};
    std::vector<uint64_t> forbidden_hashes;
    forbidden_hashes.reserve(forbidden_motifs.size());
    for (const std::string& motif : forbidden_motifs) {
        forbidden_hashes.push_back(hash_kmer_cpu(motif.c_str(), 0, static_cast<int>(motif.size())));
    }

    FeatureResult gpu_result{};
    gpu_result.gc_violations = nullptr;
    gpu_result.homopolymer_violations = nullptr;
    gpu_result.motif_collisions = nullptr;
    gpu_result.entropy_vector = nullptr;
    gpu_result.sequence_length = kSequenceLength;

    PipelineTimings timings{};

    run_feature_pipeline_async(h_sequence,
                               kSequenceLength,
                               kMotifK,
                               kEntropyK,
                               kMinEntropy,
                               forbidden_hashes,
                               &gpu_result,
                               &timings);

    std::vector<int> cpu_gc;
    std::vector<int> cpu_homo;
    std::vector<int> cpu_motif;
    std::vector<float> cpu_entropy;

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_gc_reference_chunked(h_sequence, kSequenceLength, PIPELINE_CHUNK_SIZE, cpu_gc);
    cpu_homopolymer_reference_chunked(h_sequence, kSequenceLength, PIPELINE_CHUNK_SIZE, cpu_homo);

    std::unordered_set<uint64_t> forbidden_set(forbidden_hashes.begin(), forbidden_hashes.end());
    cpu_motif_reference_chunked(
        h_sequence, kSequenceLength, PIPELINE_CHUNK_SIZE, kMotifK, forbidden_set, cpu_motif);

    cpu_entropy_reference_chunked(h_sequence, kSequenceLength, PIPELINE_CHUNK_SIZE, kEntropyK, cpu_entropy);
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    const double cpu_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(cpu_stop - cpu_start).count();

    size_t mism_gc = 0;
    size_t mism_homo = 0;
    size_t mism_motif = 0;
    size_t mism_entropy = 0;

    for (int i = 0; i < kSequenceLength; ++i) {
        mism_gc += static_cast<size_t>(gpu_result.gc_violations[i] != cpu_gc[i]);
        mism_homo += static_cast<size_t>(gpu_result.homopolymer_violations[i] != cpu_homo[i]);
        mism_motif += static_cast<size_t>(gpu_result.motif_collisions[i] != cpu_motif[i]);
        mism_entropy +=
            static_cast<size_t>(std::fabs(gpu_result.entropy_vector[i] - cpu_entropy[i]) > 1.0e-3f);
    }

    std::cout << "=== Validation ===\n";
    std::cout << "GC mismatches           : " << mism_gc << "\n";
    std::cout << "Homopolymer mismatches  : " << mism_homo << "\n";
    std::cout << "Motif mismatches        : " << mism_motif << "\n";
    std::cout << "Entropy mismatches      : " << mism_entropy << "\n";
    std::cout << "CPU reference time (s)  : " << cpu_seconds << "\n\n";

    std::cout << "=== GPU Timings ===\n";
    std::cout << "H2D transfer ms         : " << timings.h2d_ms << "\n";
    std::cout << "GC kernel ms            : " << timings.gc_ms << "\n";
    std::cout << "Homopolymer kernel ms   : " << timings.homopolymer_ms << "\n";
    std::cout << "Motif kernel ms         : " << timings.motif_ms << "\n";
    std::cout << "Entropy kernel ms       : " << timings.entropy_ms << "\n";
    std::cout << "D2H transfer ms         : " << timings.d2h_ms << "\n";
    std::cout << "Total pipeline ms       : " << timings.total_ms << "\n\n";

    const size_t input_bytes = static_cast<size_t>(kSequenceLength) * sizeof(char);

    std::cout << "=== Throughput ===\n";
    std::cout << "GC      : " << gb_per_second(input_bytes, timings.gc_ms) << " GB/s, "
              << bases_per_second(kSequenceLength, timings.gc_ms) << " bases/s\n";
    std::cout << "HOMO    : " << gb_per_second(input_bytes, timings.homopolymer_ms) << " GB/s, "
              << bases_per_second(kSequenceLength, timings.homopolymer_ms) << " bases/s\n";
    std::cout << "MOTIF   : " << gb_per_second(input_bytes, timings.motif_ms) << " GB/s, "
              << bases_per_second(kSequenceLength, timings.motif_ms) << " bases/s\n";
    std::cout << "ENTROPY : " << gb_per_second(input_bytes, timings.entropy_ms) << " GB/s, "
              << bases_per_second(kSequenceLength, timings.entropy_ms) << " bases/s\n";

    if (gpu_result.gc_violations != nullptr) {
        CUDA_CHECK(cudaFreeHost(gpu_result.gc_violations));
    }
    if (gpu_result.homopolymer_violations != nullptr) {
        CUDA_CHECK(cudaFreeHost(gpu_result.homopolymer_violations));
    }
    if (gpu_result.motif_collisions != nullptr) {
        CUDA_CHECK(cudaFreeHost(gpu_result.motif_collisions));
    }
    if (gpu_result.entropy_vector != nullptr) {
        CUDA_CHECK(cudaFreeHost(gpu_result.entropy_vector));
    }

    CUDA_CHECK(cudaFreeHost(h_sequence));

    return 0;
}
