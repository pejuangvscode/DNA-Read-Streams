# Penjelasan Detail Proyek DNA Pipeline CUDA

## 1. Proyek Ini Sebenarnya Mengerjakan Apa

Proyek ini membangun mesin analisis kualitas sekuens DNA berukuran besar dengan fokus pada kecepatan tinggi.
Secara praktis, program menerima sekuens DNA sintetis (huruf `A/C/G/T`) lalu menilai kualitasnya menggunakan beberapa metrik biologis penting.

Empat metrik utama yang dihitung:

1. GC content violation: apakah rasio `G` dan `C` di dalam window melanggar batas aman.
2. Homopolymer violation: apakah ada run karakter sama yang terlalu panjang.
3. Forbidden motif collision: apakah ada pola (motif) terlarang yang muncul.
4. Shannon entropy: apakah kompleksitas lokal sekuens cukup tinggi.

Program tidak hanya menghitung hasil fitur, tapi juga membandingkan performa:

- Jalur GPU paralel (utama, cepat).
- Jalur CPU skuensial (baseline pembanding).

## 2. Mengapa Proyek Ini Dibuat

Pada analisis DNA skala besar, bottleneck utama biasanya ada di komputasi sliding window dan hashing motif berulang.
Jika diproses serial, waktu komputasi cepat membengkak ketika panjang sekuens naik.

Karena itu proyek ini memakai CUDA untuk:

- Memecah kerja menjadi banyak thread paralel.
- Menumpuk transfer data dan komputasi (pipeline 3 stream).
- Mempertahankan validasi hasil dengan baseline CPU.

Tujuan akhirnya adalah throughput tinggi sambil tetap transparan secara akurasi.

## 3. Data Masuk dan Cara Dataset Dibuat

Dataset dibangkitkan di `src/main.cpp` secara reproducible dari file biner input:

- Panjang default: `10,000,000` basa.
- Alphabet: `A`, `C`, `G`, `T`.
- RNG: `mt19937`.
- Seed: `12345`.

Dampaknya:

- Eksperimen bisa diulang dengan hasil input yang sama.
- Benchmark antar-run lebih adil.
- Mudah dipakai untuk profiling dan regresi performa.

## 4. Alur End-to-End Eksekusi

Urutan kerja program dari awal sampai selesai:

1. Alokasi pinned host memory untuk input agar transfer async efisien.
2. Generate sekuens DNA sintetis.
3. Persiapan motif terlarang (di-hash lebih dulu).
4. Jalankan pipeline GPU asinkron berbasis chunk.
5. Jalankan CPU reference chunked.
6. Jalankan CPU baseline skuensial global.
7. Hitung mismatch GPU vs CPU.
8. Hitung timing, throughput, dan speedup.
9. Tampilkan ringkasan di terminal.
10. Simpan ringkasan ke `prof/Parallel`, `prof/Sequential`, atau `prof/Both`.
11. Bebaskan semua resource host/device.

## 5. Arsitektur Paralel: 3-Stream Pipeline

Di `pipeline.cu`, data diproses per chunk (`PIPELINE_CHUNK_SIZE`) dengan tiga stream:

1. Stream H2D: copy chunk `n+1` dari host ke GPU.
2. Stream Compute: jalankan kernel untuk chunk `n`.
3. Stream D2H: copy hasil chunk `n-1` dari GPU ke host.

Model ini membuat transfer dan komputasi overlap sehingga GPU tidak banyak idle.
Sinkronisasi antar tahap dilakukan dengan CUDA events (`cudaEventRecord`, `cudaStreamWaitEvent`).

## 6. Apa yang Dikerjakan Tiap Modul Fitur

### 6.1 GC Content (`gc_content.cu`)

Logika:

1. Tiap basa dipetakan ke biner:
   - `G/C` -> `1`
   - `A/T` -> `0`
2. Dibangun prefix sum.
3. Jumlah GC pada window dihitung O(1):
   - `gc_count = prefix[i + W] - prefix[i]`
4. Rasio GC dibandingkan terhadap `GC_MIN` dan `GC_MAX`.

Tujuan:

- Menandai area sekuens yang terlalu kaya atau terlalu miskin GC.

### 6.2 Homopolymer (`homopolymer.cu`)

Logika:

1. Bentuk sinyal apakah karakter saat ini sama dengan sebelumnya.
2. Gunakan segmented scan agar run otomatis reset saat karakter berubah.
3. Tandai violation jika panjang run > `MAX_HOMOPOLYMER`.

Tujuan:

- Menghindari run panjang karakter sama yang sering bermasalah pada proses downstream.

### 6.3 Motif Matching (`motif_match.cu`)

Logika:

1. Encode basa ke 2-bit (`A=00, C=01, G=10, T=11`).
2. Hitung rolling hash k-mer.
3. Cocokkan dengan daftar hash motif terlarang di constant memory.

Tujuan:

- Deteksi pola lokal yang dianggap tidak diinginkan.

### 6.4 Entropy (`entropy.cu`)

Logika:

1. Untuk tiap window, bentuk histogram k-mer di shared memory.
2. Hitung probabilitas tiap bin.
3. Hitung entropy Shannon:
   - `H = -sum(p * log2(p))`
4. Tandai violation jika entropy di bawah ambang minimum.

Tujuan:

- Mengukur keragaman lokal sekuens; entropy rendah berarti pola terlalu repetitif.

### 6.5 Prefix Scan (`prefix_scan.cu`)

Ini adalah primitive inti yang dipakai lintas modul:

- Exclusive scan untuk akumulasi cepat.
- Segmented scan untuk kasus yang perlu reset antar-segmen.

Tanpa primitive ini, banyak query window/range akan jauh lebih mahal.

## 7. Kenapa Ada Dua Baseline CPU

Program menjalankan dua pembanding CPU karena tujuan validasi dan interpretasi berbeda:

1. CPU reference chunked:
   - Meniru perilaku chunk pipeline GPU.
   - Cocok untuk mengecek konsistensi implementasi chunk.

2. CPU sequential global:
   - Pendekatan serial murni sepanjang sekuens.
   - Cocok untuk baseline performa dan acuan matematis global.

## 8. Kenapa Mismatch Bisa Terjadi

Mismatch tertentu bisa muncul bukan karena kernel rusak, tapi karena perbedaan model batas data:

- GPU bekerja per chunk.
- Window yang menyentuh batas chunk dapat berperilaku berbeda dari model global kontinu jika tidak ada halo overlap.

Artinya mismatch perlu dibaca bersama konteks arsitektur, bukan hanya angka mentah.

## 9. Metrik yang Dilaporkan

Program mencetak dan menyimpan metrik berikut:

1. Mismatch per fitur.
2. Timing GPU:
   - H2D, GC, homopolymer, motif, entropy, D2H, total.
3. Timing CPU sequential:
   - per fitur dan total.
4. Throughput:
   - `bases/s`
   - `GB/s`
5. Speedup total:
   - `CPU sequential total / GPU total`.

Ini membuat proyek tidak hanya "jalan", tapi juga terukur dari sisi akurasi dan performa.

## 10. Output Akhir yang Dihasilkan

Output runtime ditulis ke:

- terminal, dan
- `prof/Parallel/benchmark_report.txt`, `prof/Sequential/benchmark_report.txt`, atau `prof/Both/benchmark_report.txt`.

Dokumen ini berisi hasil validasi dan performa yang bisa langsung dipakai untuk laporan eksperimen/profiling.

## 11. Ringkasan Singkat

Secara sederhana, proyek ini adalah pipeline analisis kualitas DNA berkecepatan tinggi:

- Menghitung fitur biologis penting secara paralel di GPU.
- Menjaga validasi dengan baseline CPU.
- Mengoptimalkan overlap transfer-komputasi melalui 3-stream pipeline.
- Menyajikan hasil dalam format yang siap dievaluasi.
