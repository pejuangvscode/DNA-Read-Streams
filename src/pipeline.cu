#include "dna_features.cuh"

#include <algorithm>
#include <cstring>
#include <vector>

using namespace std;

namespace {

constexpr int kPipelineSlots = 3;

inline cudaEvent_t create_event(unsigned flags = cudaEventDefault) {
    cudaEvent_t ev = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, flags));
    return ev;
}

inline void destroy_events(vector<cudaEvent_t>& events) {
    for (cudaEvent_t ev : events) {
        if (ev != nullptr) {
            CUDA_CHECK(cudaEventDestroy(ev));
        }
    }
}

}  // namespace

void run_feature_pipeline_async(const char* h_sequence,
                                int length,
                                int motif_k,
                                int entropy_k,
                                float min_entropy,
                                const vector<uint64_t>& forbidden_hashes,
                                FeatureResult* h_result,
                                PipelineTimings* timings) {
    if (length <= 0 || h_sequence == nullptr || h_result == nullptr || timings == nullptr) {
        return;
    }

    timings->h2d_ms = 0.0f;
    timings->d2h_ms = 0.0f;
    timings->gc_ms = 0.0f;
    timings->homopolymer_ms = 0.0f;
    timings->motif_ms = 0.0f;
    timings->entropy_ms = 0.0f;
    timings->total_ms = 0.0f;

    cudaEvent_t total_start = create_event();
    cudaEvent_t total_stop = create_event();

    cudaStream_t stream_h2d = nullptr;
    cudaStream_t stream_compute = nullptr;
    cudaStream_t stream_d2h = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream_h2d));
    CUDA_CHECK(cudaStreamCreate(&stream_compute));
    CUDA_CHECK(cudaStreamCreate(&stream_d2h));

    CUDA_CHECK(cudaEventRecord(total_start, stream_compute));

    const int num_chunks = (length + PIPELINE_CHUNK_SIZE - 1) / PIPELINE_CHUNK_SIZE;
    vector<int> chunk_offsets(num_chunks);
    vector<int> chunk_lengths(num_chunks);
    for (int c = 0; c < num_chunks; ++c) {
        const int offset = c * PIPELINE_CHUNK_SIZE;
        const int len = min(PIPELINE_CHUNK_SIZE, length - offset);
        chunk_offsets[c] = offset;
        chunk_lengths[c] = len;
    }

    // Pinned host staging enables true async copies for pageable input/output buffers.
    char* h_input_stage[kPipelineSlots] = {nullptr, nullptr, nullptr};
    int* h_gc_pinned = nullptr;
    int* h_homo_pinned = nullptr;
    int* h_motif_pinned = nullptr;
    float* h_entropy_pinned = nullptr;

    for (int s = 0; s < kPipelineSlots; ++s) {
        CUDA_CHECK(cudaMallocHost(&h_input_stage[s], PIPELINE_CHUNK_SIZE * sizeof(char)));
    }

    CUDA_CHECK(cudaMallocHost(&h_gc_pinned, static_cast<size_t>(length) * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_homo_pinned, static_cast<size_t>(length) * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_motif_pinned, static_cast<size_t>(length) * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_entropy_pinned, static_cast<size_t>(length) * sizeof(float)));

    char* d_sequence[kPipelineSlots] = {nullptr, nullptr, nullptr};
    int* d_gc_viol[kPipelineSlots] = {nullptr, nullptr, nullptr};
    int* d_gc_prefix[kPipelineSlots] = {nullptr, nullptr, nullptr};
    int* d_homo_viol[kPipelineSlots] = {nullptr, nullptr, nullptr};
    int* d_run_len[kPipelineSlots] = {nullptr, nullptr, nullptr};
    int* d_motif_viol[kPipelineSlots] = {nullptr, nullptr, nullptr};
    int* d_motif_prefix[kPipelineSlots] = {nullptr, nullptr, nullptr};
    float* d_entropy[kPipelineSlots] = {nullptr, nullptr, nullptr};
    int* d_entropy_viol[kPipelineSlots] = {nullptr, nullptr, nullptr};

    for (int s = 0; s < kPipelineSlots; ++s) {
        CUDA_CHECK(cudaMalloc(&d_sequence[s], PIPELINE_CHUNK_SIZE * sizeof(char)));
        CUDA_CHECK(cudaMalloc(&d_gc_viol[s], PIPELINE_CHUNK_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_gc_prefix[s], (PIPELINE_CHUNK_SIZE + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_homo_viol[s], PIPELINE_CHUNK_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_run_len[s], PIPELINE_CHUNK_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_motif_viol[s], PIPELINE_CHUNK_SIZE * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_motif_prefix[s], (PIPELINE_CHUNK_SIZE + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_entropy[s], PIPELINE_CHUNK_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_entropy_viol[s], PIPELINE_CHUNK_SIZE * sizeof(int)));
    }

    vector<cudaEvent_t> ev_h2d_done(kPipelineSlots, nullptr);
    vector<cudaEvent_t> ev_compute_done(kPipelineSlots, nullptr);
    vector<cudaEvent_t> ev_d2h_done(kPipelineSlots, nullptr);

    for (int s = 0; s < kPipelineSlots; ++s) {
        ev_h2d_done[s] = create_event(cudaEventDisableTiming);
        ev_compute_done[s] = create_event(cudaEventDisableTiming);
        ev_d2h_done[s] = create_event(cudaEventDisableTiming);
    }

    vector<cudaEvent_t> h2d_start(num_chunks, nullptr);
    vector<cudaEvent_t> h2d_stop(num_chunks, nullptr);
    vector<cudaEvent_t> d2h_start(num_chunks, nullptr);
    vector<cudaEvent_t> d2h_stop(num_chunks, nullptr);
    vector<cudaEvent_t> gc_start(num_chunks, nullptr);
    vector<cudaEvent_t> gc_stop(num_chunks, nullptr);
    vector<cudaEvent_t> homo_start(num_chunks, nullptr);
    vector<cudaEvent_t> homo_stop(num_chunks, nullptr);
    vector<cudaEvent_t> motif_start(num_chunks, nullptr);
    vector<cudaEvent_t> motif_stop(num_chunks, nullptr);
    vector<cudaEvent_t> entropy_start(num_chunks, nullptr);
    vector<cudaEvent_t> entropy_stop(num_chunks, nullptr);

    for (int i = 0; i < num_chunks; ++i) {
        h2d_start[i] = create_event();
        h2d_stop[i] = create_event();
        d2h_start[i] = create_event();
        d2h_stop[i] = create_event();
        gc_start[i] = create_event();
        gc_stop[i] = create_event();
        homo_start[i] = create_event();
        homo_stop[i] = create_event();
        motif_start[i] = create_event();
        motif_stop[i] = create_event();
        entropy_start[i] = create_event();
        entropy_stop[i] = create_event();
    }

    if (!forbidden_hashes.empty()) {
        set_forbidden_motifs(forbidden_hashes.data(), static_cast<int>(forbidden_hashes.size()), motif_k);
    } else {
        set_forbidden_motifs(nullptr, 0, motif_k);
    }

    for (int step = 0; step < num_chunks + 2; ++step) {
        const int load_chunk = step;
        if (load_chunk < num_chunks) {
            const int slot = load_chunk % kPipelineSlots;
            const int offset = chunk_offsets[load_chunk];
            const int len = chunk_lengths[load_chunk];

            // Ensure slot is free before reusing it for the next H2D transfer.
            if (load_chunk >= kPipelineSlots) {
                CUDA_CHECK(cudaStreamWaitEvent(stream_h2d, ev_d2h_done[slot], 0));
            }

            memcpy(h_input_stage[slot], h_sequence + offset, static_cast<size_t>(len) * sizeof(char));

            CUDA_CHECK(cudaEventRecord(h2d_start[load_chunk], stream_h2d));
            CUDA_CHECK(cudaMemcpyAsync(d_sequence[slot],
                                       h_input_stage[slot],
                                       static_cast<size_t>(len) * sizeof(char),
                                       cudaMemcpyHostToDevice,
                                       stream_h2d));
            CUDA_CHECK(cudaEventRecord(h2d_stop[load_chunk], stream_h2d));
            CUDA_CHECK(cudaEventRecord(ev_h2d_done[slot], stream_h2d));
        }

        const int compute_chunk = step - 1;
        if (compute_chunk >= 0 && compute_chunk < num_chunks) {
            const int slot = compute_chunk % kPipelineSlots;
            const int len = chunk_lengths[compute_chunk];

            CUDA_CHECK(cudaStreamWaitEvent(stream_compute, ev_h2d_done[slot], 0));

            CUDA_CHECK(cudaEventRecord(gc_start[compute_chunk], stream_compute));
            run_gc_content_feature(
                d_sequence[slot], len, WINDOW_SIZE, d_gc_viol[slot], d_gc_prefix[slot], stream_compute);
            CUDA_CHECK(cudaEventRecord(gc_stop[compute_chunk], stream_compute));

            CUDA_CHECK(cudaEventRecord(homo_start[compute_chunk], stream_compute));
            run_homopolymer_feature(
                d_sequence[slot], len, d_homo_viol[slot], d_run_len[slot], stream_compute);
            CUDA_CHECK(cudaEventRecord(homo_stop[compute_chunk], stream_compute));

            CUDA_CHECK(cudaEventRecord(motif_start[compute_chunk], stream_compute));
            run_motif_feature(
                d_sequence[slot], len, motif_k, d_motif_viol[slot], d_motif_prefix[slot], stream_compute);
            CUDA_CHECK(cudaEventRecord(motif_stop[compute_chunk], stream_compute));

            CUDA_CHECK(cudaEventRecord(entropy_start[compute_chunk], stream_compute));
            run_entropy_feature(d_sequence[slot],
                                len,
                                entropy_k,
                                min_entropy,
                                d_entropy[slot],
                                d_entropy_viol[slot],
                                stream_compute);
            CUDA_CHECK(cudaEventRecord(entropy_stop[compute_chunk], stream_compute));

            CUDA_CHECK(cudaEventRecord(ev_compute_done[slot], stream_compute));
        }

        const int unload_chunk = step - 2;
        if (unload_chunk >= 0 && unload_chunk < num_chunks) {
            const int slot = unload_chunk % kPipelineSlots;
            const int offset = chunk_offsets[unload_chunk];
            const int len = chunk_lengths[unload_chunk];

            CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, ev_compute_done[slot], 0));

            CUDA_CHECK(cudaEventRecord(d2h_start[unload_chunk], stream_d2h));
            CUDA_CHECK(cudaMemcpyAsync(h_gc_pinned + offset,
                                       d_gc_viol[slot],
                                       static_cast<size_t>(len) * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream_d2h));
            CUDA_CHECK(cudaMemcpyAsync(h_homo_pinned + offset,
                                       d_homo_viol[slot],
                                       static_cast<size_t>(len) * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream_d2h));
            CUDA_CHECK(cudaMemcpyAsync(h_motif_pinned + offset,
                                       d_motif_viol[slot],
                                       static_cast<size_t>(len) * sizeof(int),
                                       cudaMemcpyDeviceToHost,
                                       stream_d2h));
            CUDA_CHECK(cudaMemcpyAsync(h_entropy_pinned + offset,
                                       d_entropy[slot],
                                       static_cast<size_t>(len) * sizeof(float),
                                       cudaMemcpyDeviceToHost,
                                       stream_d2h));
            CUDA_CHECK(cudaEventRecord(d2h_stop[unload_chunk], stream_d2h));
            CUDA_CHECK(cudaEventRecord(ev_d2h_done[slot], stream_d2h));
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_d2h));
    CUDA_CHECK(cudaEventRecord(total_stop, stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_h2d));

    for (int i = 0; i < num_chunks; ++i) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, h2d_start[i], h2d_stop[i]));
        timings->h2d_ms += ms;

        CUDA_CHECK(cudaEventElapsedTime(&ms, d2h_start[i], d2h_stop[i]));
        timings->d2h_ms += ms;

        CUDA_CHECK(cudaEventElapsedTime(&ms, gc_start[i], gc_stop[i]));
        timings->gc_ms += ms;

        CUDA_CHECK(cudaEventElapsedTime(&ms, homo_start[i], homo_stop[i]));
        timings->homopolymer_ms += ms;

        CUDA_CHECK(cudaEventElapsedTime(&ms, motif_start[i], motif_stop[i]));
        timings->motif_ms += ms;

        CUDA_CHECK(cudaEventElapsedTime(&ms, entropy_start[i], entropy_stop[i]));
        timings->entropy_ms += ms;
    }

    CUDA_CHECK(cudaEventElapsedTime(&timings->total_ms, total_start, total_stop));

    if (h_result->gc_violations == nullptr) {
        h_result->gc_violations = h_gc_pinned;
    } else {
        memcpy(h_result->gc_violations, h_gc_pinned, static_cast<size_t>(length) * sizeof(int));
        CUDA_CHECK(cudaFreeHost(h_gc_pinned));
    }

    if (h_result->homopolymer_violations == nullptr) {
        h_result->homopolymer_violations = h_homo_pinned;
    } else {
        memcpy(h_result->homopolymer_violations,
                    h_homo_pinned,
                    static_cast<size_t>(length) * sizeof(int));
        CUDA_CHECK(cudaFreeHost(h_homo_pinned));
    }

    if (h_result->motif_collisions == nullptr) {
        h_result->motif_collisions = h_motif_pinned;
    } else {
        memcpy(h_result->motif_collisions, h_motif_pinned, static_cast<size_t>(length) * sizeof(int));
        CUDA_CHECK(cudaFreeHost(h_motif_pinned));
    }

    if (h_result->entropy_vector == nullptr) {
        h_result->entropy_vector = h_entropy_pinned;
    } else {
        memcpy(h_result->entropy_vector, h_entropy_pinned, static_cast<size_t>(length) * sizeof(float));
        CUDA_CHECK(cudaFreeHost(h_entropy_pinned));
    }

    h_result->sequence_length = length;

    for (int s = 0; s < kPipelineSlots; ++s) {
        CUDA_CHECK(cudaFreeHost(h_input_stage[s]));
    }

    for (int s = 0; s < kPipelineSlots; ++s) {
        CUDA_CHECK(cudaFree(d_sequence[s]));
        CUDA_CHECK(cudaFree(d_gc_viol[s]));
        CUDA_CHECK(cudaFree(d_gc_prefix[s]));
        CUDA_CHECK(cudaFree(d_homo_viol[s]));
        CUDA_CHECK(cudaFree(d_run_len[s]));
        CUDA_CHECK(cudaFree(d_motif_viol[s]));
        CUDA_CHECK(cudaFree(d_motif_prefix[s]));
        CUDA_CHECK(cudaFree(d_entropy[s]));
        CUDA_CHECK(cudaFree(d_entropy_viol[s]));
    }

    destroy_events(ev_h2d_done);
    destroy_events(ev_compute_done);
    destroy_events(ev_d2h_done);
    destroy_events(h2d_start);
    destroy_events(h2d_stop);
    destroy_events(d2h_start);
    destroy_events(d2h_stop);
    destroy_events(gc_start);
    destroy_events(gc_stop);
    destroy_events(homo_start);
    destroy_events(homo_stop);
    destroy_events(motif_start);
    destroy_events(motif_stop);
    destroy_events(entropy_start);
    destroy_events(entropy_stop);
    CUDA_CHECK(cudaEventDestroy(total_start));
    CUDA_CHECK(cudaEventDestroy(total_stop));

    CUDA_CHECK(cudaStreamDestroy(stream_h2d));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_d2h));
}
