# GPU-Accelerated DNA Feature Extraction Pipeline (CUDA C++)

Proyek ini melakukan ekstraksi fitur kualitas sekuens DNA dengan dua pendekatan:

- Paralel GPU (CUDA, asynchronous 3-stream pipeline).
- Skuensial CPU (single-thread baseline) untuk perbandingan performa.

Fitur yang dihitung:

- GC content violation (windowed).
- Homopolymer violation (run-length).
- Forbidden motif collision (k-mer hashing).
- Shannon entropy per window.

## Struktur Proyek

- `dna_features.cuh`: konstanta, struct hasil, deklarasi API.
- `prefix_scan.cu`: Blelloch exclusive scan + segmented scan.
- `gc_content.cu`: kernel GC mapping + query berbasis prefix-scan.
- `homopolymer.cu`: kernel homopolymer berbasis segmented scan.
- `motif_match.cu`: kernel motif matching (2-bit hash, constant memory).
- `entropy.cu`: kernel entropy (shared histogram + warp reduction).
- `pipeline.cu`: orkestrasi 3 stream (`H2D -> compute -> D2H`) per chunk.
- `main.cpp`: dataset generator, benchmark, validasi, compare paralel vs sekuensial.

## Konstanta Penting

Ada di `dna_features.cuh`:

- `WINDOW_SIZE = 50`
- `MAX_HOMOPOLYMER = 3`
- `GC_MIN = 0.45f`
- `GC_MAX = 0.55f`
- `PIPELINE_CHUNK_SIZE = 1 << 20` (1 juta base per chunk)

## Build dan Run (Linux)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/dna_pipeline
```

Jika generator berbeda, binary dapat berada pada path lain di bawah `build/`.

## Gambaran Arsitektur Paralel (GPU)

Pipeline GPU memproses data dalam chunk dan menumpuk tiga tahap secara overlap:

1. Stream H2D menyalin chunk `n+1` ke device.
2. Stream compute mengeksekusi semua kernel untuk chunk `n`.
3. Stream D2H menyalin hasil chunk `n-1` ke host.

Sinkronisasi antar tahap dilakukan dengan CUDA event (`cudaEventRecord`, `cudaStreamWaitEvent`).

### Detail Tiap Fitur Paralel

1. GC Content:
   - Mapping karakter DNA menjadi biner GC (`1` untuk `G/C`, selain itu `0`).
   - Exclusive scan untuk membangun prefix sum.
   - Query window O(1):
     $$gc(i, W) = prefix[i+W] - prefix[i]$$
   - Violation jika rasio di luar $[GC_{min}, GC_{max}]$.

2. Homopolymer:
   - Bentuk sinyal `H[i] = (S[i] == S[i-1])`.
   - Segmented exclusive scan untuk reset otomatis saat run putus.
   - Violation jika panjang run melebihi `MAX_HOMOPOLYMER`.

3. Motif Matching:
   - Base DNA di-encode 2-bit: `A=00`, `C=01`, `G=10`, `T=11`.
   - Hitung hash k-mer lalu bandingkan dengan daftar hash forbidden di constant memory.

4. Entropy:
   - Tiap window membangun histogram k-mer ukuran $4^k$ di shared memory.
   - Entropy dihitung:
     $$H = -\sum_x p_x \log_2(p_x)$$
   - Window ditandai violation bila `H < min_entropy`.

## Algoritma Skuensial (CPU Baseline)

`main.cpp` kini memiliki baseline sekuensial eksplisit:

- `cpu_gc_sequential`
- `cpu_homopolymer_sequential`
- `cpu_motif_sequential`
- `cpu_entropy_sequential`
- `run_cpu_pipeline_sequential` (ukur waktu per fitur dan total)

Karakteristik baseline:

- Single-thread, tanpa CUDA.
- Dipakai sebagai pembanding performa terhadap pipeline paralel GPU.
- Menghasilkan output fitur yang bisa dibandingkan langsung dengan output GPU (mismatch report tersedia di output program).

## Metode Compare Paralel vs Skuensial

Program mencetak:

1. Validasi terhadap CPU reference chunked (emulasi perilaku chunk GPU).
2. Validasi terhadap CPU sekuensial baseline.
3. Waktu GPU:
   - H2D, GC, homopolymer, motif, entropy, D2H, total pipeline.
4. Waktu CPU sekuensial:
   - per fitur + total.
5. Speedup total:
   $$speedup = \frac{T_{cpu\_seq}}{T_{gpu\_total}}$$

Throughput dihitung sebagai:

- Bases per second:
  $$\text{bases/s} = \frac{N}{ms \times 10^{-3}}$$
- GB/s:
  $$\text{GB/s} = \frac{bytes / 10^9}{ms \times 10^{-3}}$$

## Bagaimana Dataset Digenerate

Dataset bersifat sintetis dan reproducible, dibuat di host sebelum pipeline dijalankan.

Implementasi ada di fungsi `generate_synthetic_dataset` (`main.cpp`):

- Panjang default: `10,000,000` base.
- Alphabet: `{A, C, G, T}`.
- RNG: `mt19937`.
- Seed default: `12345` (deterministik, hasil repeatable antar run pada konfigurasi sama).
- Distribusi pemilihan base: uniform diskrit pada 4 simbol.

Pseudo proses:

1. Inisialisasi RNG dengan seed tetap.
2. Untuk setiap posisi `i` dari `0..N-1`:
   - Ambil angka acak di rentang `[0,3]`.
   - Map ke `A/C/G/T`.
   - Simpan ke buffer sequence.

Ini memastikan benchmark stabil, mudah direproduksi, dan cocok untuk uji regresi performa.

## Catatan Boundary

Pipeline GPU diproses per chunk untuk throughput tinggi. Pada fitur windowed, perilaku di tepi chunk dapat berbeda dari pendekatan global kontinu jika tidak ada halo overlap. Karena itu, proyek ini menampilkan lebih dari satu mode validasi agar perbedaan perilaku terlihat eksplisit.