#include "dna_features.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

namespace {

struct CpuSequentialTimings {
    double gc_ms;
    double homopolymer_ms;
    double motif_ms;
    double entropy_ms;
    double total_ms;
};

void generate_synthetic_dataset(char* seq, int n, uint32_t seed) {
    mt19937 rng(seed);
    uniform_int_distribution<int> dist(0, 3);
    const char alphabet[4] = {'A', 'C', 'G', 'T'};
    for (int i = 0; i < n; ++i) {
        seq[i] = alphabet[dist(rng)];
    }
}

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

void cpu_gc_reference_chunked(const char* seq, int n, int chunk_size, vector<int>& out) {
    out.assign(n, 0);

    for (int offset = 0; offset < n; offset += chunk_size) {
        const int len = min(chunk_size, n - offset);
        vector<int> prefix(len + 1, 0);
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

void cpu_homopolymer_reference_chunked(const char* seq, int n, int chunk_size, vector<int>& out) {
    out.assign(n, 0);

    for (int offset = 0; offset < n; offset += chunk_size) {
        const int len = min(chunk_size, n - offset);
        vector<int> h(len, 0);
        vector<int> run(len, 0);

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
                                 const unordered_set<uint64_t>& forbidden,
                                 vector<int>& out) {
    out.assign(n, 0);

    for (int offset = 0; offset < n; offset += chunk_size) {
        const int len = min(chunk_size, n - offset);

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
                                   vector<float>& out) {
    out.assign(n, 0.0f);
    const int bins = 1 << (2 * k);

    for (int offset = 0; offset < n; offset += chunk_size) {
        const int len = min(chunk_size, n - offset);
        const int valid_windows = len - WINDOW_SIZE + 1;
        if (valid_windows <= 0) {
            continue;
        }

        vector<int> hist(bins, 0);
        for (int i = 0; i < valid_windows; ++i) {
            fill(hist.begin(), hist.end(), 0);
            const int kmers = WINDOW_SIZE - k + 1;
            for (int p = 0; p < kmers; ++p) {
                const uint64_t h = hash_kmer_cpu(seq + offset + i, p, k);
                hist[static_cast<int>(h)] += 1;
            }

            float entropy = 0.0f;
            for (int b = 0; b < bins; ++b) {
                if (hist[b] > 0) {
                    const float prob = static_cast<float>(hist[b]) / static_cast<float>(kmers);
                    entropy += -prob * log2(prob);
                }
            }
            out[offset + i] = entropy;
        }
    }
}

void cpu_gc_sequential(const char* seq, int n, vector<int>& out) {
    out.assign(n, 0);
    if (n <= 0) {
        return;
    }

    vector<int> prefix(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        const char b = seq[i];
        const int is_gc = (b == 'G' || b == 'C' || b == 'g' || b == 'c') ? 1 : 0;
        prefix[i + 1] = prefix[i] + is_gc;
    }

    for (int i = 0; i + WINDOW_SIZE <= n; ++i) {
        const int gc_count = prefix[i + WINDOW_SIZE] - prefix[i];
        const float ratio = static_cast<float>(gc_count) / static_cast<float>(WINDOW_SIZE);
        out[i] = static_cast<int>(ratio < GC_MIN || ratio > GC_MAX);
    }
}

void cpu_homopolymer_sequential(const char* seq, int n, vector<int>& out) {
    out.assign(n, 0);
    if (n <= 1) {
        return;
    }

    int run = 0;
    for (int i = 1; i < n; ++i) {
        if (seq[i] == seq[i - 1]) {
            run += 1;
        } else {
            run = 0;
        }
        out[i] = static_cast<int>(run > MAX_HOMOPOLYMER);
    }
}

void cpu_motif_sequential(const char* seq,
                         int n,
                         int k,
                         const unordered_set<uint64_t>& forbidden,
                         vector<int>& out) {
    out.assign(n, 0);
    if (k <= 0 || n < k) {
        return;
    }

    for (int i = 0; i + k <= n; ++i) {
        const uint64_t h = hash_kmer_cpu(seq, i, k);
        out[i] = static_cast<int>(forbidden.find(h) != forbidden.end());
    }
}

void cpu_entropy_sequential(const char* seq, int n, int k, vector<float>& out) {
    out.assign(n, 0.0f);
    if (k <= 0 || n <= 0) {
        return;
    }

    const int bins = 1 << (2 * k);
    const int valid_windows = n - WINDOW_SIZE + 1;
    if (valid_windows <= 0) {
        return;
    }

    vector<int> hist(bins, 0);
    for (int i = 0; i < valid_windows; ++i) {
        fill(hist.begin(), hist.end(), 0);
        const int kmers = WINDOW_SIZE - k + 1;
        for (int p = 0; p < kmers; ++p) {
            const uint64_t h = hash_kmer_cpu(seq + i, p, k);
            hist[static_cast<int>(h)] += 1;
        }

        float entropy = 0.0f;
        for (int b = 0; b < bins; ++b) {
            if (hist[b] > 0) {
                const float prob = static_cast<float>(hist[b]) / static_cast<float>(kmers);
                entropy += -prob * log2(prob);
            }
        }
        out[i] = entropy;
    }
}

void run_cpu_pipeline_sequential(const char* seq,
                                 int n,
                                 int motif_k,
                                 int entropy_k,
                                 const unordered_set<uint64_t>& forbidden,
                                 vector<int>& gc,
                                 vector<int>& homopolymer,
                                 vector<int>& motif,
                                 vector<float>& entropy,
                                 CpuSequentialTimings* timings) {
    if (timings == nullptr) {
        return;
    }

    auto total_start = chrono::high_resolution_clock::now();

    auto t0 = chrono::high_resolution_clock::now();
    cpu_gc_sequential(seq, n, gc);
    auto t1 = chrono::high_resolution_clock::now();
    timings->gc_ms = chrono::duration_cast<chrono::duration<double, milli>>(t1 - t0).count();

    t0 = chrono::high_resolution_clock::now();
    cpu_homopolymer_sequential(seq, n, homopolymer);
    t1 = chrono::high_resolution_clock::now();
    timings->homopolymer_ms = chrono::duration_cast<chrono::duration<double, milli>>(t1 - t0).count();

    t0 = chrono::high_resolution_clock::now();
    cpu_motif_sequential(seq, n, motif_k, forbidden, motif);
    t1 = chrono::high_resolution_clock::now();
    timings->motif_ms = chrono::duration_cast<chrono::duration<double, milli>>(t1 - t0).count();

    t0 = chrono::high_resolution_clock::now();
    cpu_entropy_sequential(seq, n, entropy_k, entropy);
    t1 = chrono::high_resolution_clock::now();
    timings->entropy_ms = chrono::duration_cast<chrono::duration<double, milli>>(t1 - t0).count();

    auto total_stop = chrono::high_resolution_clock::now();
    timings->total_ms =
        chrono::duration_cast<chrono::duration<double, milli>>(total_stop - total_start).count();
}

double bases_per_second(int bases, double ms) {
    if (ms <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(bases) / (static_cast<double>(ms) * 1.0e-3);
}

double gb_per_second(size_t bytes, double ms) {
    if (ms <= 0.0) {
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

    constexpr uint32_t kDatasetSeed = 12345u;
    generate_synthetic_dataset(h_sequence, kSequenceLength, kDatasetSeed);

    vector<string> forbidden_motifs = {"AAAA", "CCCC", "GGGG", "TTTT", "ACGTACGT"};
    vector<uint64_t> forbidden_hashes;
    forbidden_hashes.reserve(forbidden_motifs.size());
    for (const string& motif : forbidden_motifs) {
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

    vector<int> cpu_gc;
    vector<int> cpu_homo;
    vector<int> cpu_motif;
    vector<float> cpu_entropy;

    vector<int> cpu_seq_gc;
    vector<int> cpu_seq_homo;
    vector<int> cpu_seq_motif;
    vector<float> cpu_seq_entropy;

    CpuSequentialTimings cpu_seq_timings{};

    auto cpu_start = chrono::high_resolution_clock::now();
    cpu_gc_reference_chunked(h_sequence, kSequenceLength, PIPELINE_CHUNK_SIZE, cpu_gc);
    cpu_homopolymer_reference_chunked(h_sequence, kSequenceLength, PIPELINE_CHUNK_SIZE, cpu_homo);

    unordered_set<uint64_t> forbidden_set(forbidden_hashes.begin(), forbidden_hashes.end());
    cpu_motif_reference_chunked(
        h_sequence, kSequenceLength, PIPELINE_CHUNK_SIZE, kMotifK, forbidden_set, cpu_motif);

    cpu_entropy_reference_chunked(h_sequence, kSequenceLength, PIPELINE_CHUNK_SIZE, kEntropyK, cpu_entropy);
    auto cpu_stop = chrono::high_resolution_clock::now();
    const double cpu_seconds =
        chrono::duration_cast<chrono::duration<double>>(cpu_stop - cpu_start).count();

    run_cpu_pipeline_sequential(h_sequence,
                                kSequenceLength,
                                kMotifK,
                                kEntropyK,
                                forbidden_set,
                                cpu_seq_gc,
                                cpu_seq_homo,
                                cpu_seq_motif,
                                cpu_seq_entropy,
                                &cpu_seq_timings);

    size_t mism_gc = 0;
    size_t mism_homo = 0;
    size_t mism_motif = 0;
    size_t mism_entropy = 0;
    size_t mism_seq_gc = 0;
    size_t mism_seq_homo = 0;
    size_t mism_seq_motif = 0;
    size_t mism_seq_entropy = 0;

    for (int i = 0; i < kSequenceLength; ++i) {
        mism_gc += static_cast<size_t>(gpu_result.gc_violations[i] != cpu_gc[i]);
        mism_homo += static_cast<size_t>(gpu_result.homopolymer_violations[i] != cpu_homo[i]);
        mism_motif += static_cast<size_t>(gpu_result.motif_collisions[i] != cpu_motif[i]);
        mism_entropy +=
            static_cast<size_t>(fabs(gpu_result.entropy_vector[i] - cpu_entropy[i]) > 1.0e-3f);

        mism_seq_gc += static_cast<size_t>(gpu_result.gc_violations[i] != cpu_seq_gc[i]);
        mism_seq_homo += static_cast<size_t>(gpu_result.homopolymer_violations[i] != cpu_seq_homo[i]);
        mism_seq_motif += static_cast<size_t>(gpu_result.motif_collisions[i] != cpu_seq_motif[i]);
        mism_seq_entropy +=
            static_cast<size_t>(fabs(gpu_result.entropy_vector[i] - cpu_seq_entropy[i]) > 1.0e-3f);
    }

    const size_t input_bytes = static_cast<size_t>(kSequenceLength) * sizeof(char);

    const double total_speedup =
        (timings.total_ms > 0.0f) ? (cpu_seq_timings.total_ms / static_cast<double>(timings.total_ms)) : 0.0;

    ostringstream report;
    report << "=== Validation ===\n";
    report << "GC mismatches           : " << mism_gc << "\n";
    report << "Homopolymer mismatches  : " << mism_homo << "\n";
    report << "Motif mismatches        : " << mism_motif << "\n";
    report << "Entropy mismatches      : " << mism_entropy << "\n";
    report << "CPU reference time (s)  : " << cpu_seconds << "\n\n";

    report << "=== Validation vs CPU Sequential ===\n";
    report << "GC mismatches           : " << mism_seq_gc << "\n";
    report << "Homopolymer mismatches  : " << mism_seq_homo << "\n";
    report << "Motif mismatches        : " << mism_seq_motif << "\n";
    report << "Entropy mismatches      : " << mism_seq_entropy << "\n\n";

    report << "=== GPU Timings ===\n";
    report << "H2D transfer ms         : " << timings.h2d_ms << "\n";
    report << "GC kernel ms            : " << timings.gc_ms << "\n";
    report << "Homopolymer kernel ms   : " << timings.homopolymer_ms << "\n";
    report << "Motif kernel ms         : " << timings.motif_ms << "\n";
    report << "Entropy kernel ms       : " << timings.entropy_ms << "\n";
    report << "D2H transfer ms         : " << timings.d2h_ms << "\n";
    report << "Total pipeline ms       : " << timings.total_ms << "\n\n";

    report << "=== CPU Sequential Timings ===\n";
    report << "GC sequential ms        : " << cpu_seq_timings.gc_ms << "\n";
    report << "Homopolymer seq ms      : " << cpu_seq_timings.homopolymer_ms << "\n";
    report << "Motif sequential ms     : " << cpu_seq_timings.motif_ms << "\n";
    report << "Entropy sequential ms   : " << cpu_seq_timings.entropy_ms << "\n";
    report << "Total sequential ms     : " << cpu_seq_timings.total_ms << "\n\n";

    report << "=== Throughput ===\n";
    report << "GC      : " << gb_per_second(input_bytes, timings.gc_ms) << " GB/s, "
           << bases_per_second(kSequenceLength, timings.gc_ms) << " bases/s\n";
    report << "HOMO    : " << gb_per_second(input_bytes, timings.homopolymer_ms) << " GB/s, "
           << bases_per_second(kSequenceLength, timings.homopolymer_ms) << " bases/s\n";
    report << "MOTIF   : " << gb_per_second(input_bytes, timings.motif_ms) << " GB/s, "
           << bases_per_second(kSequenceLength, timings.motif_ms) << " bases/s\n";
    report << "ENTROPY : " << gb_per_second(input_bytes, timings.entropy_ms) << " GB/s, "
           << bases_per_second(kSequenceLength, timings.entropy_ms) << " bases/s\n";

    report << "\n=== Compare Parallel vs Sequential ===\n";
    report << "Speedup total (CPU seq / GPU total) : " << total_speedup << "x\n";
    report << "CPU sequential throughput           : "
           << gb_per_second(input_bytes, cpu_seq_timings.total_ms) << " GB/s, "
           << bases_per_second(kSequenceLength, cpu_seq_timings.total_ms) << " bases/s\n";

    report << "\n=== Dataset Generation ===\n";
    report << "Dataset type             : synthetic random DNA\n";
    report << "Sequence length          : " << kSequenceLength << " bases\n";
    report << "Alphabet                 : A, C, G, T\n";
    report << "RNG                      : mt19937\n";
    report << "Seed                     : " << kDatasetSeed << "\n";

    const string report_text = report.str();
    cout << report_text;

    const string output_dir = "output";
    const string report_path = output_dir + "/benchmark_report.txt";
    std::error_code fs_error;
    filesystem::create_directories(output_dir, fs_error);

    if (!fs_error) {
        ofstream report_file(report_path, ios::out | ios::trunc);
        if (report_file) {
            report_file << report_text;
            cout << "\nReport saved to: " << report_path << "\n";
        } else {
            cerr << "\nFailed to write report file: " << report_path << "\n";
        }
    } else {
        cerr << "\nFailed to create output directory: " << fs_error.message() << "\n";
    }

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
