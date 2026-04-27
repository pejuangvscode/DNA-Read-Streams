# Konteks Riset: GPU-Accelerated DNA Quality Feature Extraction Pipeline

## 1. Ringkasan Eksekutif

Proyek ini mengimplementasikan pipeline analisis kualitas sekuens DNA berukuran besar dengan akselerasi GPU (CUDA), lalu membandingkan performa dan konsistensinya terhadap baseline CPU sequential. Fokus utama proyek adalah meningkatkan throughput komputasi untuk fitur-fitur lokal berbasis window tanpa mengorbankan keterlacakan hasil.

Empat fitur utama yang dianalisis:

1. GC content violation
2. Homopolymer violation
3. Forbidden motif collision
4. Shannon entropy per window

Pipeline dirancang untuk volume data besar (default 10 juta basa), dengan arsitektur chunked asynchronous 3-stream agar transfer data dan komputasi dapat overlap.

## 2. Latar Belakang Masalah

Pada domain bioinformatics dan DNA data processing, kualitas sekuens sering ditentukan oleh metrik lokal yang harus dihitung berulang pada window yang saling overlap. Pendekatan serial murni menjadi mahal saat ukuran data membesar karena:

1. Banyak operasi berulang pada range yang hampir sama.
2. Kebutuhan hash dan histogram pada tiap window.
3. Bottleneck transfer bila komputasi dipindah ke GPU tanpa orkestrasi yang tepat.

Masalah riset yang dijawab proyek ini adalah bagaimana membangun pipeline GPU berthroughput tinggi untuk quality-feature extraction yang:

1. Dapat diprofiling secara jelas.
2. Memiliki baseline pembanding yang eksplisit.
3. Tetap memungkinkan validasi hasil lintas mode eksekusi.

## 3. Tujuan dan Pertanyaan Riset

### 3.1 Tujuan

1. Mendesain pipeline paralel untuk ekstraksi fitur DNA pada data skala besar.
2. Menurunkan biaya komputasi query window melalui scan-based methods.
3. Mengukur speedup terhadap baseline CPU sequential.
4. Menyediakan proses profiling yang bersih dan reproducible.

### 3.2 Pertanyaan Riset

1. Seberapa besar peningkatan throughput dari arsitektur 3-stream dibanding sequential?
2. Fitur mana yang paling dominan terhadap total waktu eksekusi?
3. Bagaimana dampak chunk boundary terhadap mismatch terhadap baseline global?
4. Sejauh mana desain pemisahan mode eksekusi membantu kualitas profiling?

## 4. Kontribusi Teknis Proyek

Kontribusi utama yang dapat diklaim:

1. Pipeline multi-feature DNA quality extraction berbasis CUDA dengan overlap H2D-compute-D2H.
2. Implementasi scan-centric acceleration untuk query range dan segmented behavior.
3. Baseline CPU sequential dan reference chunked untuk validasi berlapis.
4. Mekanisme mode eksekusi terpisah (parallel, sequential, both) plus binary terpisah untuk profiling bersih.
5. Input dataset generik dari file biner, dipetakan naif ke alfabet ACGT menggunakan skema 2-bit.

## 5. Definisi Input dan Representasi Data

### 5.1 Sumber Data

Input berasal dari file biner apa pun melalui argumen input-file.

### 5.2 Pemetaan Biner ke DNA

Setiap byte diproses sebagai empat simbol 2-bit dengan urutan bit:

1. bit 7..6
2. bit 5..4
3. bit 3..2
4. bit 1..0

Mapping simbol:

1. 00 -> A
2. 01 -> C
3. 10 -> G
4. 11 -> T

Jika panjang sekuens target melebihi data file, bytes diputar ulang (cyclic replay) hingga panjang target tercapai.

## 6. Arsitektur Sistem

### 6.1 Komponen Kode

1. [src/main.cpp](../src/main.cpp): orchestration mode run, dataset generation, benchmark, reporting
2. [src/pipeline.cu](../src/pipeline.cu): chunk pipeline 3-stream + CUDA events
3. [src/gc_content.cu](../src/gc_content.cu): GC map + prefix-scan query
4. [src/homopolymer.cu](../src/homopolymer.cu): run-length logic via segmented scan
5. [src/motif_match.cu](../src/motif_match.cu): 2-bit rolling hash motif matching
6. [src/entropy.cu](../src/entropy.cu): shared histogram + entropy reduction
7. [src/prefix_scan.cu](../src/prefix_scan.cu): exclusive scan and segmented variants
8. [src/dna_features.cuh](../src/dna_features.cuh): constants, data structures, declarations
9. [CMakeLists.txt](../CMakeLists.txt): build targets default, parallel-only, sequential-only

### 6.2 Mode Eksekusi

1. both: GPU + reference chunked + CPU sequential + perbandingan
2. parallel: hanya jalur GPU
3. sequential: hanya jalur CPU sequential

### 6.3 Pemisahan Binary

1. dna_pipeline
2. dna_pipeline_parallel
3. dna_pipeline_sequential

Pemisahan ini meminimalkan kontaminasi profiling ketika hanya mode tertentu yang diinginkan.

## 7. Metodologi Algoritmik

### 7.1 GC Content

Langkah:

1. Map setiap basa menjadi biner GC: G/C = 1, lainnya 0.
2. Bangun prefix sum eksklusif.
3. Query jumlah GC di window:

gc_count(i, W) = prefix(i + W) - prefix(i)

4. Hitung rasio dan tandai pelanggaran jika rasio di luar rentang threshold.

Keuntungan utama adalah query window O(1) setelah preprocessing O(n).

### 7.2 Homopolymer

Langkah:

1. Bentuk sinyal kesamaan terhadap elemen sebelumnya.
2. Jalankan segmented exclusive scan agar run reset otomatis saat simbol berubah.
3. Hitung run-length efektif dan beri label violation jika melewati ambang maksimum.

### 7.3 Forbidden Motif

Langkah:

1. Encode ACGT menjadi representasi 2-bit.
2. Hitung rolling hash k-mer untuk setiap posisi valid.
3. Cocokkan hash dengan daftar forbidden motif (constant memory di device).

### 7.4 Entropy

Langkah:

1. Untuk tiap window, bentuk histogram k-mer (4 pangkat k bins) di shared memory.
2. Ubah frekuensi menjadi probabilitas.
3. Hitung entropy Shannon:

H = - sum p(x) log2 p(x)

4. Flag violation jika entropy di bawah ambang minimum.

## 8. Pipeline Paralel dan Overlap

Pada chunk ke-n:

1. H2D memindahkan chunk n+1
2. Compute mengeksekusi kernel chunk n
3. D2H memindahkan hasil chunk n-1

Koordinasi dilakukan melalui CUDA events agar slot buffer aman dipakai ulang.

Implikasi:

1. Transfer dan compute dapat overlap.
2. Total wall-time menurun dibanding eksekusi sinkron berurutan.

## 9. Baseline dan Strategi Validasi

Digunakan dua baseline CPU:

1. Reference chunked
2. Sequential global

Tujuan:

1. Reference chunked menguji konsistensi desain chunking GPU.
2. Sequential global menjadi acuan serial murni untuk speedup dan analisis mismatch.

Interpretasi mismatch:

1. Mismatch pada fitur windowed dapat dipicu boundary effect lintas chunk.
2. Hal ini perlu dibedakan dari bug implementasi.

## 10. Kompleksitas dan Karakteristik Performa

Secara umum:

1. Scan preprocessing linear terhadap panjang data.
2. Query range menjadi konstan per posisi untuk fitur tertentu.
3. Entropy cenderung paling mahal karena histogram per-window dan reduksi.

Pada profiling praktis, kernel entropy biasanya menjadi hotspot dominan.

## 11. Setup Eksperimen yang Direkomendasikan

### 11.1 Build

1. CMake release build.
2. Gunakan arsitektur CUDA sesuai GPU yang tersedia.

### 11.2 Eksekusi

1. Parallel-only untuk profil GPU murni.
2. Sequential-only untuk baseline serial.
3. Both untuk validasi silang.

### 11.3 Profiling

Gunakan Nsight Systems pada binary parallel-only agar trace tidak bercampur dengan workload sequential.

Output profiling disimpan ke folder `prof/Parallel`.

## 12. Metrik Evaluasi

Metrik utama yang dapat dipakai dalam paper:

1. Total runtime per mode
2. Runtime per tahap GPU (H2D, kernels, D2H)
3. Throughput bases per second
4. Throughput GB per second
5. Speedup sequential over parallel
6. Mismatch count per fitur

## 13. Reproducibility dan Artefak

Elemen reproducibility yang sudah ada:

1. CLI mode run yang eksplisit
2. Input-file berbasis file biner
3. Laporan otomatis ke output folder terpisah
4. Binary terpisah untuk parallel dan sequential

Dokumen pendukung:

1. [README.md](../README.md)
2. [docs/ALUR_PROGRAM.md](ALUR_PROGRAM.md)
3. [docs/PENJELASAN_DETAIL_PROYEK.md](PENJELASAN_DETAIL_PROYEK.md)
4. [docs/MASALAH_GAP_SOLUSI_PROYEK.md](MASALAH_GAP_SOLUSI_PROYEK.md)

## 14. Ancaman terhadap Validitas

1. Input mapping biner ke ACGT bersifat naif, belum merepresentasikan distribusi biologis nyata.
2. Efek boundary chunk dapat memengaruhi komparasi terhadap baseline global.
3. Hasil performa bergantung pada arsitektur GPU, driver, dan konfigurasi sistem.
4. Profiling dengan beban non-isolated dapat bias jika mode tidak dipisah.

## 15. Peluang Pengembangan Menjadi Paper Lebih Kuat

1. Tambahkan eksperimen multi-dataset (synthetic, real sequencing-like distributions).
2. Tambahkan studi sensitivitas parameter (window size, motif k, entropy k, chunk size).
3. Tambahkan ablation study:
   - tanpa overlap stream
   - tanpa scan optimization
   - tanpa constant memory motif
4. Tambahkan analisis energy-performance (jika alat ukur tersedia).
5. Tambahkan baseline tambahan (OpenMP CPU, single-stream GPU, library scan standar).

## 16. Ringkasan untuk Abstrak Kandidat

Proyek ini menunjukkan bahwa ekstraksi fitur kualitas DNA dapat dipercepat signifikan melalui desain pipeline CUDA berbasis chunk dan overlap 3-stream, dengan validasi berlapis terhadap baseline CPU. Pendekatan scan-centric menurunkan biaya query window, sementara pemisahan mode eksekusi dan binary memperjelas proses profiling serta reproducibility eksperimen. Kerangka ini layak dijadikan fondasi empirical study untuk paper yang membahas trade-off akurasi, throughput, dan desain arsitektur paralel pada analisis sekuens skala besar.
