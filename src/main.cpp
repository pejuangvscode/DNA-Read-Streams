#include "dna_features.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;

namespace {

enum class RunMode {
    Both,
    ParallelOnly,
    SequentialOnly,
};

RunMode parse_run_mode(int argc, char** argv) {
#if defined(DNA_FORCE_PARALLEL_MODE)
    (void)argc;
    (void)argv;
    return RunMode::ParallelOnly;
#elif defined(DNA_FORCE_SEQUENTIAL_MODE)
    (void)argc;
    (void)argv;
    return RunMode::SequentialOnly;
#else
    RunMode mode = RunMode::Both;

    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        if (arg == "--parallel") {
            mode = RunMode::ParallelOnly;
        } else if (arg == "--sequential") {
            mode = RunMode::SequentialOnly;
        } else if (arg == "--both") {
            mode = RunMode::Both;
        } else if (arg.rfind("--mode=", 0) == 0) {
            const string value = arg.substr(7);
            if (value == "parallel") {
                mode = RunMode::ParallelOnly;
            } else if (value == "sequential") {
                mode = RunMode::SequentialOnly;
            } else if (value == "both") {
                mode = RunMode::Both;
            }
        } else if (arg == "--mode" && i + 1 < argc) {
            const string value = argv[++i];
            if (value == "parallel") {
                mode = RunMode::ParallelOnly;
            } else if (value == "sequential") {
                mode = RunMode::SequentialOnly;
            } else if (value == "both") {
                mode = RunMode::Both;
            }
        }
    }

    return mode;
#endif
}

const char* run_mode_to_string(RunMode mode) {
    switch (mode) {
        case RunMode::ParallelOnly:
            return "parallel";
        case RunMode::SequentialOnly:
            return "sequential";
        case RunMode::Both:
        default:
            return "both";
    }
}

struct CpuSequentialTimings {
    double gc_ms;
    double homopolymer_ms;
    double motif_ms;
    double entropy_ms;
    double total_ms;
};

string parse_input_file_path(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        if (arg.rfind("--input-file=", 0) == 0) {
            return arg.substr(13);
        }
        if (arg == "--input-file" && i + 1 < argc) {
            return argv[i + 1];
        }
    }
    return "";
}

int parse_sequence_length(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        if (arg.rfind("--sequence-length=", 0) == 0) {
            return stoi(arg.substr(18));
        }
        if (arg == "--sequence-length" && i + 1 < argc) {
            return stoi(argv[i + 1]);
        }
    }
    return 0;
}

vector<int> parse_sizes(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        string csv;
        if (arg.rfind("--sizes=", 0) == 0) {
            csv = arg.substr(8);
        } else if (arg == "--sizes" && i + 1 < argc) {
            csv = argv[++i];
        }
        if (!csv.empty()) {
            vector<int> sizes;
            istringstream ss(csv);
            string token;
            while (getline(ss, token, ',')) {
                try {
                    const int v = stoi(token);
                    if (v > 0) sizes.push_back(v);
                } catch (...) {}
            }
            sort(sizes.begin(), sizes.end());
            sizes.erase(unique(sizes.begin(), sizes.end()), sizes.end());
            return sizes;
        }
    }
    return {};
}

struct BenchmarkRecord {
    int    sequence_length = 0;
    double gpu_h2d_ms      = 0.0;
    double gpu_gc_ms       = 0.0;
    double gpu_homo_ms     = 0.0;
    double gpu_motif_ms    = 0.0;
    double gpu_entropy_ms  = 0.0;
    double gpu_d2h_ms      = 0.0;
    double gpu_total_ms    = 0.0;
    double cpu_gc_ms       = 0.0;
    double cpu_homo_ms     = 0.0;
    double cpu_motif_ms    = 0.0;
    double cpu_entropy_ms  = 0.0;
    double cpu_total_ms    = 0.0;
    double speedup         = 0.0;
    double gpu_bases_per_s = 0.0;
    double cpu_bases_per_s = 0.0;
    double gpu_gbs         = 0.0;
    double cpu_gbs         = 0.0;
    double gpu_gflops      = 0.0;
    double cpu_gflops      = 0.0;
};

double estimate_gflops(int n, int motif_k, int entropy_k, double ms) {
    if (ms <= 0.0 || n <= 0) return 0.0;
    // GC: N float divisions + N comparisons
    const double gc_flops = static_cast<double>(n) * 2.0;
    // Homopolymer: N comparisons
    const double homo_flops = static_cast<double>(n) * 1.0;
    // Motif: N * motif_k 2-bit encodes + N comparisons
    const double motif_flops = static_cast<double>(n) * (static_cast<double>(motif_k) + 1.0) * 0.5;
    // Entropy: per window -> bins * log2 + warp reduce
    const int bins = 1 << (2 * entropy_k);
    const int valid_windows = max(0, n - WINDOW_SIZE + 1);
    const double entropy_flops = static_cast<double>(valid_windows) * (static_cast<double>(bins) * 3.0 + 32.0);
    const double total_flops = gc_flops + homo_flops + motif_flops + entropy_flops;
    return (total_flops / 1.0e9) / (ms * 1.0e-3);
}

void write_benchmark_csv(const vector<BenchmarkRecord>& records, const string& path) {
    filesystem::create_directories(filesystem::path(path).parent_path());
    ofstream f(path, ios::out | ios::trunc);
    if (!f) { cerr << "Failed to write CSV: " << path << "\n"; return; }
    f << "sequence_length,"
      << "gpu_h2d_ms,gpu_gc_ms,gpu_homo_ms,gpu_motif_ms,gpu_entropy_ms,gpu_d2h_ms,gpu_total_ms,"
      << "cpu_gc_ms,cpu_homo_ms,cpu_motif_ms,cpu_entropy_ms,cpu_total_ms,"
      << "speedup,gpu_bases_per_s,cpu_bases_per_s,gpu_gbs,cpu_gbs,gpu_gflops,cpu_gflops\n";
    for (const auto& r : records) {
        f << r.sequence_length << ","
          << r.gpu_h2d_ms << "," << r.gpu_gc_ms << "," << r.gpu_homo_ms << ","
          << r.gpu_motif_ms << "," << r.gpu_entropy_ms << "," << r.gpu_d2h_ms << ","
          << r.gpu_total_ms << ","
          << r.cpu_gc_ms << "," << r.cpu_homo_ms << "," << r.cpu_motif_ms << ","
          << r.cpu_entropy_ms << "," << r.cpu_total_ms << ","
          << r.speedup << ","
          << r.gpu_bases_per_s << "," << r.cpu_bases_per_s << ","
          << r.gpu_gbs << "," << r.cpu_gbs << ","
          << r.gpu_gflops << "," << r.cpu_gflops << "\n";
    }
}

vector<unsigned char> read_binary_input_file(const string& path) {
    if (path.empty()) {
        throw runtime_error("missing --input-file argument");
    }

    ifstream input(path, ios::binary);
    if (!input) {
        throw runtime_error("failed to open input file: " + path);
    }

    vector<unsigned char> bytes((istreambuf_iterator<char>(input)), istreambuf_iterator<char>());
    if (bytes.empty()) {
        throw runtime_error("input file is empty: " + path);
    }

    return bytes;
}

int resolve_sequence_length(const vector<unsigned char>& bytes, int sequence_length_override) {
    if (sequence_length_override > 0) {
        return sequence_length_override;
    }

    const uint64_t derived_length = static_cast<uint64_t>(bytes.size()) * 4ull;
    if (derived_length == 0ull || derived_length > static_cast<uint64_t>(numeric_limits<int>::max())) {
        throw runtime_error("derived sequence length is invalid or too large; use --sequence-length");
    }

    return static_cast<int>(derived_length);
}

void generate_dataset_from_binary_bytes(const vector<unsigned char>& bytes, char* seq, int n) {
    const char alphabet[4] = {'A', 'C', 'G', 'T'};
    for (int i = 0; i < n; ++i) {
        const size_t byte_idx = static_cast<size_t>(i / 4) % bytes.size();
        const int two_bit_pos = i % 4;
        const int shift = 6 - (two_bit_pos * 2);
        const unsigned char value = static_cast<unsigned char>((bytes[byte_idx] >> shift) & 0x03u);
        seq[i] = alphabet[value];
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

int main(int argc, char** argv) {
    const RunMode run_mode = parse_run_mode(argc, argv);
    const bool run_parallel = (run_mode != RunMode::SequentialOnly);
    const bool run_sequential = (run_mode != RunMode::ParallelOnly);
    const bool run_chunked_reference = (run_mode == RunMode::Both);

    constexpr int kMotifK = 8;
    constexpr int kEntropyK = 3;
    constexpr float kMinEntropy = 1.60f;
    const string input_file_path = parse_input_file_path(argc, argv);
    const int sequence_length_override = parse_sequence_length(argc, argv);
    const vector<int> benchmark_sizes = parse_sizes(argc, argv);
    const bool is_benchmark_mode = !benchmark_sizes.empty();

    vector<unsigned char> input_file_bytes;
    int kSequenceLength = 0;

    try {
        input_file_bytes = read_binary_input_file(input_file_path);
        if (is_benchmark_mode) {
            // Allocate for the largest requested size; smaller sizes reuse the same buffer.
            kSequenceLength = benchmark_sizes.back();
        } else {
            kSequenceLength = resolve_sequence_length(input_file_bytes, sequence_length_override);
        }
    } catch (const exception& e) {
        cerr << "Input preparation error: " << e.what() << "\n";
        cerr << "Usage: " << argv[0]
             << " --input-file <path> [--sequence-length <n>] [--mode parallel|sequential|both]\n";
        return 1;
    }

    // Use pinned memory only when GPU is needed (faster H2D transfers).
    // For sequential-only mode, plain malloc avoids a hard CUDA dependency.
    char* h_sequence = nullptr;
    const bool need_gpu = run_parallel || is_benchmark_mode;
    bool h_sequence_is_pinned = false;

    if (need_gpu) {
        cudaError_t pin_err = cudaMallocHost(&h_sequence,
                                             static_cast<size_t>(kSequenceLength) * sizeof(char));
        if (pin_err == cudaSuccess) {
            h_sequence_is_pinned = true;
        } else {
            // GPU not available or out of pinned memory — abort if GPU is truly required.
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,
                    cudaGetErrorString(pin_err));
            return 1;
        }
    } else {
        // Sequential-only: plain malloc, no CUDA dependency.
        h_sequence = static_cast<char*>(malloc(static_cast<size_t>(kSequenceLength) * sizeof(char)));
        if (!h_sequence) {
            cerr << "Out of memory allocating sequence buffer.\n";
            return 1;
        }
    }

    auto free_h_sequence = [&]() {
        if (h_sequence_is_pinned) { cudaFreeHost(h_sequence); }
        else                      { free(h_sequence); }
        h_sequence = nullptr;
    };

    try {
        generate_dataset_from_binary_bytes(input_file_bytes, h_sequence, kSequenceLength);
    } catch (const exception& e) {
        cerr << "Dataset generation error: " << e.what() << "\n";
        cerr << "Usage: " << argv[0]
             << " --input-file <path> [--sequence-length <n>] [--mode parallel|sequential|both]\n";
        free_h_sequence();
        return 1;
    }

    unordered_set<uint64_t> forbidden_set_early;

    vector<string> forbidden_motifs = {"AAAA", "CCCC", "GGGG", "TTTT", "ACGTACGT"};
    vector<uint64_t> forbidden_hashes;
    forbidden_hashes.reserve(forbidden_motifs.size());
    for (const string& motif : forbidden_motifs) {
        forbidden_hashes.push_back(hash_kmer_cpu(motif.c_str(), 0, static_cast<int>(motif.size())));
    }

    for (const auto& h : forbidden_hashes) forbidden_set_early.insert(h);

    // ── BENCHMARK SWEEP MODE ─────────────────────────────────────────────────
    if (is_benchmark_mode) {
        cout << "\n=== Benchmark Sweep Mode ==="
             << " (" << benchmark_sizes.size() << " sizes)\n";
        cout << "Sizes: ";
        for (int sz : benchmark_sizes) cout << sz << " ";
        cout << "\n\n";

        vector<BenchmarkRecord> records;
        records.reserve(benchmark_sizes.size());

        for (int sz : benchmark_sizes) {
            const size_t sz_bytes = static_cast<size_t>(sz);
            cout << "[Sweep] N = " << sz << " bases ... " << flush;

            BenchmarkRecord rec;
            rec.sequence_length = sz;

            // ── GPU parallel ──────────────────────────────────────────────
            FeatureResult gpu_res{};
            gpu_res.gc_violations          = nullptr;
            gpu_res.homopolymer_violations = nullptr;
            gpu_res.motif_collisions       = nullptr;
            gpu_res.entropy_vector         = nullptr;
            gpu_res.sequence_length        = sz;
            PipelineTimings pt{};

            run_feature_pipeline_async(h_sequence, sz, kMotifK, kEntropyK,
                                       kMinEntropy, forbidden_hashes, &gpu_res, &pt);

            rec.gpu_h2d_ms     = static_cast<double>(pt.h2d_ms);
            rec.gpu_gc_ms      = static_cast<double>(pt.gc_ms);
            rec.gpu_homo_ms    = static_cast<double>(pt.homopolymer_ms);
            rec.gpu_motif_ms   = static_cast<double>(pt.motif_ms);
            rec.gpu_entropy_ms = static_cast<double>(pt.entropy_ms);
            rec.gpu_d2h_ms     = static_cast<double>(pt.d2h_ms);
            rec.gpu_total_ms   = static_cast<double>(pt.total_ms);
            rec.gpu_bases_per_s = bases_per_second(sz, static_cast<double>(pt.total_ms));
            rec.gpu_gbs         = gb_per_second(sz_bytes, static_cast<double>(pt.total_ms));
            rec.gpu_gflops      = estimate_gflops(sz, kMotifK, kEntropyK,
                                                  static_cast<double>(pt.total_ms));

            if (gpu_res.gc_violations)          CUDA_CHECK(cudaFreeHost(gpu_res.gc_violations));
            if (gpu_res.homopolymer_violations) CUDA_CHECK(cudaFreeHost(gpu_res.homopolymer_violations));
            if (gpu_res.motif_collisions)       CUDA_CHECK(cudaFreeHost(gpu_res.motif_collisions));
            if (gpu_res.entropy_vector)         CUDA_CHECK(cudaFreeHost(gpu_res.entropy_vector));

            // ── CPU sequential ────────────────────────────────────────────
            vector<int>   seq_gc, seq_homo, seq_motif;
            vector<float> seq_entropy;
            CpuSequentialTimings ct{};

            run_cpu_pipeline_sequential(h_sequence, sz, kMotifK, kEntropyK,
                                        forbidden_set_early,
                                        seq_gc, seq_homo, seq_motif, seq_entropy, &ct);

            rec.cpu_gc_ms       = ct.gc_ms;
            rec.cpu_homo_ms     = ct.homopolymer_ms;
            rec.cpu_motif_ms    = ct.motif_ms;
            rec.cpu_entropy_ms  = ct.entropy_ms;
            rec.cpu_total_ms    = ct.total_ms;
            rec.cpu_bases_per_s = bases_per_second(sz, ct.total_ms);
            rec.cpu_gbs         = gb_per_second(sz_bytes, ct.total_ms);
            rec.cpu_gflops      = estimate_gflops(sz, kMotifK, kEntropyK, ct.total_ms);

            rec.speedup = (pt.total_ms > 0.0f)
                              ? (ct.total_ms / static_cast<double>(pt.total_ms))
                              : 0.0;

            cout << "GPU=" << rec.gpu_total_ms << "ms"
                 << " | CPU=" << rec.cpu_total_ms << "ms"
                 << " | Speedup=" << rec.speedup << "x"
                 << " | GPU_GFLOPS=" << rec.gpu_gflops << "\n";

            records.push_back(rec);
        }

        const string csv_path = "prof/benchmark_sweep.csv";
        write_benchmark_csv(records, csv_path);
        cout << "\nBenchmark CSV saved to: " << csv_path << "\n";

        // Print summary table
        cout << "\n" << string(90, '-') << "\n";
        cout << "  N (bases)   | GPU total ms | CPU total ms | Speedup  | GPU GFLOPS | CPU GFLOPS\n";
        cout << string(90, '-') << "\n";
        for (const auto& r : records) {
            cout << "  " << setw(11) << r.sequence_length
                 << " | " << setw(12) << r.gpu_total_ms
                 << " | " << setw(12) << r.cpu_total_ms
                 << " | " << setw(8)  << r.speedup
                 << " | " << setw(10) << r.gpu_gflops
                 << " | " << setw(10) << r.cpu_gflops << "\n";
        }
        cout << string(90, '-') << "\n";

        free_h_sequence();
        return 0;
    }
    // ── END BENCHMARK SWEEP MODE ─────────────────────────────────────────────

    FeatureResult gpu_result{};
    gpu_result.gc_violations = nullptr;
    gpu_result.homopolymer_violations = nullptr;
    gpu_result.motif_collisions = nullptr;
    gpu_result.entropy_vector = nullptr;
    gpu_result.sequence_length = kSequenceLength;

    PipelineTimings timings{};

    if (run_parallel) {
        run_feature_pipeline_async(h_sequence,
                                   kSequenceLength,
                                   kMotifK,
                                   kEntropyK,
                                   kMinEntropy,
                                   forbidden_hashes,
                                   &gpu_result,
                                   &timings);
    }

    vector<int> cpu_gc;
    vector<int> cpu_homo;
    vector<int> cpu_motif;
    vector<float> cpu_entropy;

    vector<int> cpu_seq_gc;
    vector<int> cpu_seq_homo;
    vector<int> cpu_seq_motif;
    vector<float> cpu_seq_entropy;

    CpuSequentialTimings cpu_seq_timings{};

    unordered_set<uint64_t> forbidden_set(forbidden_hashes.begin(), forbidden_hashes.end());
    forbidden_set.insert(forbidden_set_early.begin(), forbidden_set_early.end());

    double cpu_seconds = 0.0;
    if (run_chunked_reference) {
        auto cpu_start = chrono::high_resolution_clock::now();
        cpu_gc_reference_chunked(h_sequence, kSequenceLength, PIPELINE_CHUNK_SIZE, cpu_gc);
        cpu_homopolymer_reference_chunked(h_sequence, kSequenceLength, PIPELINE_CHUNK_SIZE, cpu_homo);
        cpu_motif_reference_chunked(
            h_sequence, kSequenceLength, PIPELINE_CHUNK_SIZE, kMotifK, forbidden_set, cpu_motif);
        cpu_entropy_reference_chunked(
            h_sequence, kSequenceLength, PIPELINE_CHUNK_SIZE, kEntropyK, cpu_entropy);
        auto cpu_stop = chrono::high_resolution_clock::now();
        cpu_seconds = chrono::duration_cast<chrono::duration<double>>(cpu_stop - cpu_start).count();
    }

    if (run_sequential) {
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
    }

    size_t mism_gc = 0;
    size_t mism_homo = 0;
    size_t mism_motif = 0;
    size_t mism_entropy = 0;
    size_t mism_seq_gc = 0;
    size_t mism_seq_homo = 0;
    size_t mism_seq_motif = 0;
    size_t mism_seq_entropy = 0;

    for (int i = 0; i < kSequenceLength; ++i) {
        if (run_parallel && run_chunked_reference) {
            mism_gc += static_cast<size_t>(gpu_result.gc_violations[i] != cpu_gc[i]);
            mism_homo += static_cast<size_t>(gpu_result.homopolymer_violations[i] != cpu_homo[i]);
            mism_motif += static_cast<size_t>(gpu_result.motif_collisions[i] != cpu_motif[i]);
            mism_entropy +=
                static_cast<size_t>(fabs(gpu_result.entropy_vector[i] - cpu_entropy[i]) > 1.0e-3f);
        }

        if (run_parallel && run_sequential) {
            mism_seq_gc += static_cast<size_t>(gpu_result.gc_violations[i] != cpu_seq_gc[i]);
            mism_seq_homo +=
                static_cast<size_t>(gpu_result.homopolymer_violations[i] != cpu_seq_homo[i]);
            mism_seq_motif += static_cast<size_t>(gpu_result.motif_collisions[i] != cpu_seq_motif[i]);
            mism_seq_entropy +=
                static_cast<size_t>(fabs(gpu_result.entropy_vector[i] - cpu_seq_entropy[i]) > 1.0e-3f);
        }
    }

    const size_t input_bytes_count = static_cast<size_t>(kSequenceLength) * sizeof(char);

        const double total_speedup =
         (timings.total_ms > 0.0f) ? (cpu_seq_timings.total_ms / static_cast<double>(timings.total_ms)) : 0.0;

    ostringstream report;
        report << "=== Run Configuration ===\n";
        report << "Mode                     : " << run_mode_to_string(run_mode) << "\n";
        report << "Use --mode parallel|sequential|both\n\n";

        if (run_parallel && run_chunked_reference) {
         report << "=== Validation ===\n";
         report << "GC mismatches           : " << mism_gc << "\n";
         report << "Homopolymer mismatches  : " << mism_homo << "\n";
         report << "Motif mismatches        : " << mism_motif << "\n";
         report << "Entropy mismatches      : " << mism_entropy << "\n";
         report << "CPU reference time (s)  : " << cpu_seconds << "\n\n";
        }

        if (run_parallel && run_sequential) {
         report << "=== Validation vs CPU Sequential ===\n";
         report << "GC mismatches           : " << mism_seq_gc << "\n";
         report << "Homopolymer mismatches  : " << mism_seq_homo << "\n";
         report << "Motif mismatches        : " << mism_seq_motif << "\n";
         report << "Entropy mismatches      : " << mism_seq_entropy << "\n\n";
        }

        if (run_parallel) {
         report << "=== GPU Timings ===\n";
         report << "H2D transfer ms         : " << timings.h2d_ms << "\n";
         report << "GC kernel ms            : " << timings.gc_ms << "\n";
         report << "Homopolymer kernel ms   : " << timings.homopolymer_ms << "\n";
         report << "Motif kernel ms         : " << timings.motif_ms << "\n";
         report << "Entropy kernel ms       : " << timings.entropy_ms << "\n";
         report << "D2H transfer ms         : " << timings.d2h_ms << "\n";
         report << "Total pipeline ms       : " << timings.total_ms << "\n\n";

         report << "=== GPU Throughput ===\n";
         report << "GC      : " << gb_per_second(input_bytes_count, timings.gc_ms) << " GB/s, "
             << bases_per_second(kSequenceLength, timings.gc_ms) << " bases/s\n";
         report << "HOMO    : " << gb_per_second(input_bytes_count, timings.homopolymer_ms) << " GB/s, "
             << bases_per_second(kSequenceLength, timings.homopolymer_ms) << " bases/s\n";
         report << "MOTIF   : " << gb_per_second(input_bytes_count, timings.motif_ms) << " GB/s, "
             << bases_per_second(kSequenceLength, timings.motif_ms) << " bases/s\n";
         report << "ENTROPY : " << gb_per_second(input_bytes_count, timings.entropy_ms) << " GB/s, "
             << bases_per_second(kSequenceLength, timings.entropy_ms) << " bases/s\n\n";
        }

        if (run_sequential) {
         report << "=== CPU Sequential Timings ===\n";
         report << "GC sequential ms        : " << cpu_seq_timings.gc_ms << "\n";
         report << "Homopolymer seq ms      : " << cpu_seq_timings.homopolymer_ms << "\n";
         report << "Motif sequential ms     : " << cpu_seq_timings.motif_ms << "\n";
         report << "Entropy sequential ms   : " << cpu_seq_timings.entropy_ms << "\n";
         report << "Total sequential ms     : " << cpu_seq_timings.total_ms << "\n";
         report << "CPU sequential throughput: "
             << gb_per_second(input_bytes_count, cpu_seq_timings.total_ms) << " GB/s, "
             << bases_per_second(kSequenceLength, cpu_seq_timings.total_ms) << " bases/s\n\n";
        }

        if (run_parallel && run_sequential) {
         report << "=== Compare Parallel vs Sequential ===\n";
         report << "Speedup total (CPU seq / GPU total) : " << total_speedup << "x\n\n";
        }

    report << "\n=== Dataset Generation ===\n";
    report << "Dataset type             : binary-file to ACGT (naive 2-bit mapping)\n";
    report << "Input file               : " << input_file_path << "\n";
    report << "Sequence length          : " << kSequenceLength << " bases\n";
    report << "Mapping                  : 00->A, 01->C, 10->G, 11->T\n";

    const string report_text = report.str();
    cout << report_text;

    string output_dir = "prof/Both";
    if (run_mode == RunMode::ParallelOnly) {
        output_dir = "prof/Parallel";
    } else if (run_mode == RunMode::SequentialOnly) {
        output_dir = "prof/Sequential";
    }
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

    if (run_parallel && gpu_result.gc_violations != nullptr) {
        CUDA_CHECK(cudaFreeHost(gpu_result.gc_violations));
    }
    if (run_parallel && gpu_result.homopolymer_violations != nullptr) {
        CUDA_CHECK(cudaFreeHost(gpu_result.homopolymer_violations));
    }
    if (run_parallel && gpu_result.motif_collisions != nullptr) {
        CUDA_CHECK(cudaFreeHost(gpu_result.motif_collisions));
    }
    if (run_parallel && gpu_result.entropy_vector != nullptr) {
        CUDA_CHECK(cudaFreeHost(gpu_result.entropy_vector));
    }

    free_h_sequence();

    return 0;
}
