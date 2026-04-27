# Alur Program, Cara Kerja, dan Algoritma

Dokumen ini menjelaskan bagaimana program berjalan dari awal sampai akhir, bagaimana modul saling terhubung, serta ringkasan algoritma yang dipakai pada jalur paralel (GPU) dan skuensial (CPU).

## 1. Tujuan Program

Program mengekstrak fitur kualitas DNA dari sequence sintetis menggunakan dua pendekatan:

- Paralel GPU (CUDA pipeline).
- Skuensial CPU (baseline pembanding).

Fitur yang dihitung:

- GC content violation.
- Homopolymer violation.
- Forbidden motif collision.
- Shannon entropy per window.

## 2. Komponen Utama

- `src/main.cpp`
  - Menghasilkan dataset sintetis.
  - Menjalankan pipeline GPU.
  - Menjalankan baseline CPU (chunked reference dan skuensial global).
  - Membandingkan hasil.
  - Menulis report ke folder `prof/` sesuai mode run.
- `src/pipeline.cu`
  - Orkestrasi 3 stream CUDA untuk overlap transfer dan komputasi.
- `src/gc_content.cu`
  - Kernel mapping GC + query window berbasis prefix scan.
- `src/homopolymer.cu`
  - Kernel run-length berbasis segmented scan.
- `src/motif_match.cu`
  - Kernel rolling hash 2-bit dan lookup motif terlarang.
- `src/entropy.cu`
  - Kernel histogram shared memory + hitung entropy.
- `src/prefix_scan.cu`
  - Primitive scan (exclusive + segmented) untuk modul lain.
- `src/dna_features.cuh`
  - Konstanta, struct output, dan deklarasi API.

## 3. Alur Eksekusi End-to-End

1. Program mengalokasikan host memory (pinned) untuk sequence.
2. Dataset sintetis dibuat dengan RNG `mt19937` dan seed tetap `12345`.
3. Motif terlarang di-hash untuk dipakai pada fitur motif.
4. Pipeline GPU dijalankan:
   - Input dibagi chunk (`PIPELINE_CHUNK_SIZE`).
   - Stream H2D mengirim chunk berikutnya.
   - Stream compute menjalankan semua kernel fitur untuk chunk aktif.
   - Stream D2H mengambil hasil chunk sebelumnya.
5. Program menjalankan validasi CPU:
   - CPU reference chunked (meniru batas chunk GPU).
   - CPU skuensial global (baseline murni satu alur).
6. Program menghitung mismatch GPU vs CPU.
7. Program menghitung metrik performa (timing, throughput, speedup).
8. Program mencetak report ke terminal.
9. Program menyimpan report yang sama ke `prof/Parallel`, `prof/Sequential`, atau `prof/Both`.
10. Program membebaskan semua resource host/device.

## 4. Dataset: Bagaimana Digenerate

Dataset bersifat sintetis, reproducible, dan uniform terhadap alfabet DNA.

- Panjang default: `10,000,000` base.
- Alfabet: `A`, `C`, `G`, `T`.
- RNG: `mt19937`.
- Seed: `12345`.
- Distribusi: uniform integer `[0, 3]`, lalu dipetakan ke alfabet.

## 5. Alur Paralel (GPU)

### 5.1 Orkestrasi 3-Stream

Untuk setiap langkah pipeline:

- `load_chunk`: copy host -> device untuk chunk ke-`n+1`.
- `compute_chunk`: jalankan kernel fitur untuk chunk ke-`n`.
- `unload_chunk`: copy device -> host untuk chunk ke-`n-1`.

Sinkronisasi antar tahap memakai CUDA event agar overlap aman dan tidak saling menimpa slot buffer.

### 5.2 Algoritma Per Fitur (GPU)

1. GC Content
- Setiap base diubah ke biner: GC = 1, non-GC = 0.
- Exclusive prefix scan membangun akumulasi cepat.
- Jumlah GC dalam window didapat O(1):
  - `gc_count = prefix[i + W] - prefix[i]`
- Violation jika rasio di luar ambang `GC_MIN..GC_MAX`.

2. Homopolymer
- Bentuk sinyal `H[i] = (S[i] == S[i-1])`.
- Gunakan segmented exclusive scan agar run reset otomatis saat base berubah.
- Tandai violation saat run length melebihi `MAX_HOMOPOLYMER`.

3. Motif Matching
- Encode DNA ke 2-bit (`A=00, C=01, G=10, T=11`).
- Hitung hash k-mer per posisi.
- Bandingkan dengan daftar hash forbidden di constant memory.

4. Entropy
- Tiap window membangun histogram k-mer (`4^k`) di shared memory.
- Hitung probabilitas tiap bin, lalu entropy Shannon:
  - `H = -sum(p * log2(p))`
- Violation jika entropy di bawah ambang minimum.

## 6. Alur Skuensial (CPU Baseline)

Baseline skuensial berjalan satu alur penuh tanpa overlap:
- `cpu_gc_sequential`
- `cpu_homopolymer_sequential`
- `cpu_motif_sequential`
- `cpu_entropy_sequential`

Lalu diorkestrasi oleh `run_cpu_pipeline_sequential` untuk mendapatkan timing per fitur dan total.

## 7. Validasi dan Perbandingan

Program menampilkan dua jenis mismatch:

- GPU vs CPU reference chunked.
- GPU vs CPU skuensial global.

Mengapa mismatch bisa muncul pada mode global:

- Pipeline GPU berbasis chunk, sehingga perilaku batas window di tepi chunk dapat berbeda dari pendekatan global kontinu.

## 8. Metrik yang Dilaporkan

- Timing GPU:
  - H2D, GC, homopolymer, motif, entropy, D2H, total.
- Timing CPU skuensial:
  - per fitur dan total.
- Throughput:
  - `bases/s = N / (ms * 1e-3)`
  - `GB/s = (bytes / 1e9) / (ms * 1e-3)`
- Speedup total:
  - `speedup = T_cpu_sequential / T_gpu_total`

## 9. Urutan Data dan Resource

1. Alokasi host pinned memory.
2. Alokasi buffer device per slot pipeline.
3. Eksekusi pipeline + event synchronization.
4. Copy hasil ke host.
5. Agregasi report.
6. Simpan report ke file output.
7. Free semua host/device resource.

## 10. Ringkasan Praktis

- Jika fokus performa: lihat bagian GPU timings dan speedup.
- Jika fokus akurasi: lihat mismatch terhadap kedua baseline.
- Jika mismatch global ingin diperkecil: tambahkan teknik halo overlap antar chunk.
