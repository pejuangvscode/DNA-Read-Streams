# GPU-Accelerated DNA Feature Extraction Pipeline (CUDA C++)

This project implements a high-throughput, sliding-window feature extraction pipeline for DNA data storage quality analysis on NVIDIA GPUs.

It includes:
- Work-efficient Blelloch prefix scan (exclusive)
- GC content profiling
- Homopolymer run-length detection via segmented scan
- Forbidden motif matching via 2-bit k-mer hashing
- Shannon entropy estimation with shared-memory histograms
- 3-stream asynchronous H2D / compute / D2H pipeline orchestration

## Project Layout

- `dna_features.cuh`: Shared constants, types, API declarations, CUDA error-check macro
- `prefix_scan.cu`: Blelloch scan kernels (templated operators, recursive multi-block scan)
- `gc_content.cu`: GC binary map + prefix-scan-based window query
- `homopolymer.cu`: Segmented scan pipeline for run-length and threshold violations
- `motif_match.cu`: 2-bit motif hashing and constant-memory motif lookup
- `entropy.cu`: Shared histogram + warp-reduced entropy per window
- `pipeline.cu`: 3-stream async chunk pipeline with events and pinned memory
- `main.cpp`: 10M-base synthetic benchmark + CPU reference validation + throughput reporting
- `CMakeLists.txt`: CUDA 12+ friendly CMake build configuration

## Key Constants

Defined in `dna_features.cuh`:

- `WINDOW_SIZE = 50`
- `MAX_HOMOPOLYMER = 3`
- `GC_MIN = 0.45f`
- `GC_MAX = 0.55f`
- `NUM_BANKS = 32`
- `NUM_BANK_BITS = 5`
- `PIPELINE_CHUNK_SIZE = 1 << 20` (1M bases)
- `CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANK_BITS)`

## Build Requirements

- CMake >= 3.20
- CUDA Toolkit (12+ recommended)
- A GPU supporting one of the configured architectures:
  - `sm_70`
  - `sm_80`
  - `sm_86`
  - `sm_90`

## Build

From project root:

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run

```powershell
./build/Release/dna_pipeline.exe
```

(Depending on generator/platform, binary may also be under `build/`.)

## What the Program Does

1. Generates a random DNA sequence of 10,000,000 bases on host.
2. Runs the GPU pipeline across chunked input using three CUDA streams:
   - Stream A: async Host->Device transfer for chunk `n+1`
   - Stream B: all feature kernels for chunk `n`
   - Stream C: async Device->Host transfer for chunk `n-1`
3. Runs CPU reference implementations for all features.
4. Compares GPU and CPU outputs (mismatch counts).
5. Prints timing and throughput metrics per kernel.

## Feature Pipeline Details

### 1) Prefix Scan (Blelloch)

- Exclusive scan with up-sweep and down-sweep phases
- Uses shared memory with bank-conflict padding
- Recursive decomposition for arrays larger than one scan block
- Supports:
  - `int` and `float` additive scans
  - segmented behavior through a templated operator path

### 2) GC Content

- Maps DNA chars to binary GC flags (`G`/`C` => 1)
- Uses vectorized `uint4` loads where aligned/full-vector path is available
- Exclusive scan enables O(1) window query:
  - `gc_count = prefix[i + W] - prefix[i]`
- Flags violation when GC ratio is outside `[0.45, 0.55]`

### 3) Homopolymer Detection

- Builds `H[i] = (S[i] == S[i-1])` with `H[0]=0`
- Segmented scan resets at run boundaries (`H[i]==0`)
- Produces run-length-like signal and flags positions where run length > 3
- Reset behavior is encoded branchlessly in operator logic

### 4) Forbidden Motif Matching

- DNA base encoding: `A=00, C=01, G=10, T=11`
- k-mer integer hash for `k` up to 20
- Forbidden motif hashes loaded into constant memory
- Match vector is scanned for O(1) range-collision queries

### 5) Shannon Entropy

- Per-window, per-block shared-memory histogram of size `4^k`
- Uses atomics in shared memory for local accumulation
- Converts frequencies to probabilities and computes:
  - `H = -sum(p_x * log2(p_x))`
- Uses warp-level reduction (`__shfl_down_sync`) for entropy aggregation
- Flags windows below configurable entropy threshold

## Validation and Output

The executable prints:

- Validation mismatches for:
  - GC
  - homopolymer
  - motif
  - entropy
- CPU reference total runtime
- GPU timings:
  - H2D
  - each kernel stage
  - D2H
  - total pipeline
- Throughput per feature in:
  - GB/s
  - bases/s

## Throughput Formula

- Bases/s: `bases / (ms * 1e-3)`
- GB/s: `(bytes / 1e9) / (ms * 1e-3)`

## Notes

- The implementation is chunk-based for throughput and overlap.
- Boundary handling across chunk edges is currently local to each chunk for windowed features. If strict global-window continuity across chunk boundaries is required, add halo overlap and cross-chunk reconciliation.

## Troubleshooting

- If CMake cannot find CUDA:
  - Ensure CUDA Toolkit is installed
  - Ensure `nvcc` and CUDA paths are visible to your shell
- If architecture mismatch occurs:
  - Update `CUDA_ARCHITECTURES` in `CMakeLists.txt` to match your GPU
- If entropy kernel runs out of shared memory:
  - Reduce `entropy_k` (histogram bins are `4^k`)

## License

Add your preferred license for distribution/use.
