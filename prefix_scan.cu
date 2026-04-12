#include "dna_features.cuh"

#include <cstdint>

namespace {

constexpr int kScanThreads = 1024;
constexpr int kScanElementsPerBlock = kScanThreads * 2;

struct SegmentedValue {
    int sum;
    int has_head;
};

template <typename T>
struct AddOp {
    __device__ __host__ inline T identity() const { return static_cast<T>(0); }
    __device__ __host__ inline T apply(const T& a, const T& b) const { return a + b; }
};

struct SegmentedResetOp {
    __device__ __host__ inline SegmentedValue identity() const { return SegmentedValue{0, 0}; }

    __device__ __host__ inline SegmentedValue apply(const SegmentedValue& a,
                                                    const SegmentedValue& b) const {
        // Branchless segmented combine:
        // If b has a head, keep b.sum. Otherwise add a.sum + b.sum.
        const int mask = -b.has_head;
        const int keep_b = b.sum & mask;
        const int add_ab = (a.sum + b.sum) & ~mask;
        return SegmentedValue{keep_b | add_ab, a.has_head | b.has_head};
    }
};

template <typename T, typename Op>
__global__ void blelloch_scan_block_kernel(const T* __restrict__ d_in,
                                           T* __restrict__ d_out,
                                           T* __restrict__ d_block_sums,
                                           int n,
                                           Op op) {
    extern __shared__ unsigned char s_raw[];
    T* s_data = reinterpret_cast<T*>(s_raw);

    const int tid = threadIdx.x;
    const int block_base = blockIdx.x * kScanElementsPerBlock;

    const int ai = tid;
    const int bi = tid + blockDim.x;

    const int bank_offset_a = CONFLICT_FREE_OFFSET(ai);
    const int bank_offset_b = CONFLICT_FREE_OFFSET(bi);

    const int gidx_a = block_base + ai;
    const int gidx_b = block_base + bi;

    s_data[ai + bank_offset_a] = (gidx_a < n) ? d_in[gidx_a] : op.identity();
    s_data[bi + bank_offset_b] = (gidx_b < n) ? d_in[gidx_b] : op.identity();

    // Up-sweep (reduce) phase.
#pragma unroll
    for (int offset = 1; offset < kScanElementsPerBlock; offset <<= 1) {
        __syncthreads();
        const int idx = ((tid + 1) * offset * 2) - 1;
        if (idx < kScanElementsPerBlock) {
            const int left = idx - offset;
            const int right = idx;
            const int left_idx = left + CONFLICT_FREE_OFFSET(left);
            const int right_idx = right + CONFLICT_FREE_OFFSET(right);
            s_data[right_idx] = op.apply(s_data[left_idx], s_data[right_idx]);
        }
    }

    __syncthreads();
    if (tid == 0) {
        const int last = kScanElementsPerBlock - 1;
        const int last_idx = last + CONFLICT_FREE_OFFSET(last);
        if (d_block_sums != nullptr) {
            d_block_sums[blockIdx.x] = s_data[last_idx];
        }
        s_data[last_idx] = op.identity();
    }

    // Down-sweep phase.
#pragma unroll
    for (int offset = kScanElementsPerBlock >> 1; offset > 0; offset >>= 1) {
        __syncthreads();
        const int idx = ((tid + 1) * offset * 2) - 1;
        if (idx < kScanElementsPerBlock) {
            const int left = idx - offset;
            const int right = idx;
            const int left_idx = left + CONFLICT_FREE_OFFSET(left);
            const int right_idx = right + CONFLICT_FREE_OFFSET(right);

            const T t = s_data[left_idx];
            s_data[left_idx] = s_data[right_idx];
            s_data[right_idx] = op.apply(t, s_data[right_idx]);
        }
    }
    __syncthreads();

    if (gidx_a < n) {
        d_out[gidx_a] = s_data[ai + bank_offset_a];
    }
    if (gidx_b < n) {
        d_out[gidx_b] = s_data[bi + bank_offset_b];
    }
}

template <typename T, typename Op>
__global__ void add_block_offsets_kernel(T* __restrict__ d_out,
                                         const T* __restrict__ d_scanned_block_sums,
                                         int n,
                                         Op op) {
    const int base = blockIdx.x * kScanElementsPerBlock;
    const int idx_a = base + threadIdx.x;
    const int idx_b = idx_a + blockDim.x;

    const T offset = d_scanned_block_sums[blockIdx.x];

    if (idx_a < n) {
        d_out[idx_a] = op.apply(offset, d_out[idx_a]);
    }
    if (idx_b < n) {
        d_out[idx_b] = op.apply(offset, d_out[idx_b]);
    }
}

__global__ void pack_segmented_values_kernel(const int* __restrict__ d_values,
                                             const int* __restrict__ d_head_flags,
                                             SegmentedValue* __restrict__ d_packed,
                                             int n) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < n) {
        d_packed[idx] = SegmentedValue{d_values[idx], d_head_flags[idx]};
    }
}

__global__ void unpack_segmented_values_kernel(const SegmentedValue* __restrict__ d_packed,
                                               int* __restrict__ d_out,
                                               int n) {
    const int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_packed[idx].sum;
    }
}

template <typename T, typename Op>
void recursive_exclusive_scan(const T* d_in, T* d_out, int n, Op op, cudaStream_t stream) {
    if (n <= 0) {
        return;
    }

    const int num_blocks = (n + kScanElementsPerBlock - 1) / kScanElementsPerBlock;
    const size_t shmem_elems = kScanElementsPerBlock + CONFLICT_FREE_OFFSET(kScanElementsPerBlock) + 1;
    const size_t shmem_bytes = shmem_elems * sizeof(T);

    T* d_block_sums = nullptr;
    T* d_scanned_block_sums = nullptr;

    if (num_blocks > 1) {
        CUDA_CHECK(cudaMalloc(&d_block_sums, static_cast<size_t>(num_blocks) * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_scanned_block_sums, static_cast<size_t>(num_blocks) * sizeof(T)));
    }

    blelloch_scan_block_kernel<T, Op>
        <<<num_blocks, kScanThreads, shmem_bytes, stream>>>(d_in, d_out, d_block_sums, n, op);
    CUDA_CHECK(cudaGetLastError());

    if (num_blocks > 1) {
        recursive_exclusive_scan(d_block_sums, d_scanned_block_sums, num_blocks, op, stream);

        add_block_offsets_kernel<T, Op>
            <<<num_blocks, kScanThreads, 0, stream>>>(d_out, d_scanned_block_sums, n, op);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_block_sums));
        CUDA_CHECK(cudaFree(d_scanned_block_sums));
    }
}

}  // namespace

void launch_exclusive_scan_int(const int* d_in, int* d_out, int n, cudaStream_t stream) {
    recursive_exclusive_scan<int, AddOp<int>>(d_in, d_out, n, AddOp<int>{}, stream);
}

void launch_exclusive_scan_float(const float* d_in, float* d_out, int n, cudaStream_t stream) {
    recursive_exclusive_scan<float, AddOp<float>>(d_in, d_out, n, AddOp<float>{}, stream);
}

void launch_segmented_exclusive_scan_int(const int* d_values,
                                         const int* d_head_flags,
                                         int* d_out,
                                         int n,
                                         cudaStream_t stream) {
    if (n <= 0) {
        return;
    }

    SegmentedValue* d_packed_in = nullptr;
    SegmentedValue* d_packed_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_packed_in, static_cast<size_t>(n) * sizeof(SegmentedValue)));
    CUDA_CHECK(cudaMalloc(&d_packed_out, static_cast<size_t>(n) * sizeof(SegmentedValue)));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    pack_segmented_values_kernel<<<blocks, threads, 0, stream>>>(d_values, d_head_flags, d_packed_in, n);
    CUDA_CHECK(cudaGetLastError());

    recursive_exclusive_scan<SegmentedValue, SegmentedResetOp>(
        d_packed_in, d_packed_out, n, SegmentedResetOp{}, stream);

    unpack_segmented_values_kernel<<<blocks, threads, 0, stream>>>(d_packed_out, d_out, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_packed_in));
    CUDA_CHECK(cudaFree(d_packed_out));
}
