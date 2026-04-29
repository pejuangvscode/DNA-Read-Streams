Kelompok 2:
- Fiolita Chresia Putri (01082230027)
- Jennifer Christabelle (01082230009)
- Gebrina Augustine Padatu (01082230041)
- Teofilus Satria Rada Insani (01082230015)


# GPU-Accelerated DNA Feature Extraction Pipeline (CUDA C++)

Proyek ini melakukan ekstraksi fitur kualitas sekuens DNA dengan dua pendekatan:

- Paralel GPU (CUDA, asynchronous 3-stream pipeline).
- Skuensial CPU (single-thread baseline) untuk perbandingan performa.

Fitur yang dihitung:

- GC content violation (windowed).
- Homopolymer violation (run-length).
- Forbidden motif collision (k-mer hashing).
- Shannon entropy per window.

Dataset ACGT sekarang berasal dari file biner input (bukan RNG random), dengan mapping 2-bit naif:

- `00 -> A`
- `01 -> C`
- `10 -> G`
- `11 -> T`

## Struktur Proyek

- `src/dna_features.cuh`: konstanta, struct hasil, deklarasi API.
- `src/prefix_scan.cu`: Blelloch exclusive scan + segmented scan.
- `src/gc_content.cu`: kernel GC mapping + query berbasis prefix-scan.
- `src/homopolymer.cu`: kernel homopolymer berbasis segmented scan.
- `src/motif_match.cu`: kernel motif matching (2-bit hash, constant memory).
- `src/entropy.cu`: kernel entropy (shared histogram + warp reduction).
- `src/pipeline.cu`: orkestrasi 3 stream (`H2D -> compute -> D2H`) per chunk.
- `src/main.cpp`: dataset generator, benchmark, validasi, compare paralel vs sekuensial.
- `docs/`: kumpulan dokumentasi naratif proyek.
- `dataset/`: tempat file input biner untuk dataset ACGT.
- `prof/Parallel` dan `prof/Sequential`: hasil benchmark/profiling yang dipisah.

## Konstanta Penting

Ada di `src/dna_features.cuh`:

- `WINDOW_SIZE = 50`
- `MAX_HOMOPOLYMER = 3`
- `GC_MIN = 0.45f`
- `GC_MAX = 0.55f`
- `PIPELINE_CHUNK_SIZE = 1 << 20` (1 juta base per chunk)

## Build (Linux)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Binary yang Tersedia

Setelah build, ada tiga binary:

- `./build/dna_pipeline`
   - Binary umum, bisa dipilih dengan mode runtime.
- `./build/dna_pipeline_parallel`
   - Binary khusus workload GPU (untuk profiling parallel yang bersih).
- `./build/dna_pipeline_sequential`
   - Binary khusus workload CPU sequential.

Jika generator berbeda, lokasi executable bisa sedikit berbeda di bawah folder `build/`.

## Cara Menjalankan

### 1) Mode runtime pada binary umum

```bash
./build/dna_pipeline --input-file dataset/dongeng.txt --mode both
./build/dna_pipeline --input-file dataset/dongeng.txt --mode parallel
./build/dna_pipeline --input-file dataset/dongeng.txt --mode sequential
```

dataset/dongeng.txt

Alias yang didukung:

```bash
./build/dna_pipeline --input-file <path_file_biner> --both
./build/dna_pipeline --input-file <path_file_biner> --parallel
./build/dna_pipeline --input-file <path_file_biner> --sequential
```

### 2) Binary terpisah

```bash
./build/dna_pipeline_parallel --input-file <path_file_biner>
./build/dna_pipeline_sequential --input-file <path_file_biner>
```

Catatan:

- Argumen `--input-file` wajib diberikan.
- Secara default, panjang sekuens otomatis mengikuti ukuran file: `jumlah_byte_file * 4` bases.
- Anda bisa override panjang sekuens dengan `--sequence-length <n>`.
- Jika `--sequence-length` lebih besar daripada data input, bytes file akan diputar ulang (cyclic).

Contoh override panjang sekuens:

```bash
./build/dna_pipeline --input-file dataset/sample_dataset-1.txt --sequence-length 1000000 --mode both
```

## Struktur Output Report

Report sekarang dipisah agar hasil tidak tercampur:

- `prof/Parallel/benchmark_report.txt`
- `prof/Sequential/benchmark_report.txt`
- `prof/Both/benchmark_report.txt` (jika menjalankan mode `both`)

Ini sengaja dipisah supaya profiling mode parallel tidak tercampur workload sequential.

## Profiling Nsight Systems (Disarankan)

Untuk profiling GPU yang bersih, jalankan binary parallel-only:

```bash
mkdir -p prof/Parallel
nsys profile \
   --force-overwrite=true \
   --trace=cuda,nvtx,osrt \
   --sample=none \
   --stats=true \
   -o prof/Parallel/dna_pipeline_parallel \
   ./build/dna_pipeline_parallel --input-file <path_file_biner>
```

Hasil profiling akan tersimpan di:

- `prof/Parallel/dna_pipeline_parallel.nsys-rep`
- `prof/Parallel/dna_pipeline_parallel.sqlite`

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

`src/main.cpp` kini memiliki baseline sekuensial eksplisit:

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

Jika mode `both` dipakai, program mencetak:

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

Dataset sekarang dibuat dari file biner eksternal yang diberikan lewat `--input-file`.

Implementasi ada di fungsi `generate_dataset_from_binary_file` (`src/main.cpp`):

- Program membaca seluruh byte file input.
- Setiap byte dipecah menjadi 4 simbol DNA (tiap simbol 2-bit):
   - bit 7..6 -> simbol 1
   - bit 5..4 -> simbol 2
   - bit 3..2 -> simbol 3
   - bit 1..0 -> simbol 4
- Mapping simbol:
   - `00 -> A`
   - `01 -> C`
   - `10 -> G`
   - `11 -> T`
- Jika panjang sekuens target lebih besar daripada data yang tersedia, byte file diputar ulang (cyclic).

Keuntungan pendekatan ini:

- Input dapat berasal dari file apa pun.
- Eksperimen bisa diuji pada pola data non-random.
- Konsisten untuk perbandingan mode `parallel`, `sequential`, dan `both` selama file input sama.

## Catatan Boundary

Pipeline GPU diproses per chunk untuk throughput tinggi. Pada fitur windowed, perilaku di tepi chunk dapat berbeda dari pendekatan global kontinu jika tidak ada halo overlap. Karena itu, proyek ini menampilkan lebih dari satu mode validasi agar perbedaan perilaku terlihat eksplisit.

## Dokumen Tambahan

- `docs/ALUR_PROGRAM.md`: ringkasan alur eksekusi dan hubungan antarmodul.
- `docs/PENJELASAN_DETAIL_PROYEK.md`: penjelasan naratif detail tentang apa yang dikerjakan proyek.