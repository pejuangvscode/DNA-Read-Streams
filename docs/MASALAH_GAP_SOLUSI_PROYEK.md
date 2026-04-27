# Masalah, Gap, Solusi, dan Implementasi Proyek

## 1. Masalah yang Ingin Diselesaikan

Analisis kualitas sekuens DNA pada skala besar (jutaan basa) cenderung lambat jika diproses secara serial, terutama untuk operasi yang berulang seperti:

- Sliding window GC content.
- Deteksi homopolymer (run karakter sama).
- Pencocokan forbidden motif berbasis k-mer.
- Perhitungan entropy lokal.

Masalah utama:

1. Waktu komputasi tinggi pada CPU serial.
2. Banyak operasi scan/range query yang mahal jika dihitung ulang naif.
3. Transfer data host-device bisa menjadi bottleneck jika tidak dioverlap.
4. Sulit memastikan akurasi saat optimasi performa dilakukan.

## 2. Gap (Kesenjangan) dari Pendekatan Naif

Jika memakai pendekatan naif:

1. Per window harus menghitung ulang dari nol, kompleksitas membengkak.
2. CPU single-thread tidak cukup efisien untuk throughput tinggi.
3. GPU tanpa orkestrasi stream hanya memindahkan bottleneck ke transfer data.
4. Tidak ada baseline validasi berlapis, sehingga sulit membedakan bug vs efek desain.

Gap teknis yang dijawab proyek ini:

- Gap performa: serial CPU vs paralel GPU.
- Gap arsitektur: eksekusi sinkron vs pipeline async 3 stream.
- Gap validasi: hasil cepat vs hasil benar (akurasi tetap terukur).
- Gap usability eksperimen: output bercampur vs output terpisah per mode.

## 3. Solusi yang Dipilih

Proyek ini mengadopsi solusi berlapis:

### A. Solusi Algoritmik

1. Prefix scan (exclusive) untuk mempercepat query range (mis. GC window jadi O(1) per query).
2. Segmented scan untuk homopolymer agar reset run efisien.
3. Rolling hash 2-bit untuk motif matching.
4. Histogram shared memory + reduksi warp untuk entropy.

### B. Solusi Arsitektur Paralel

1. Pemrosesan berbasis chunk.
2. Pipeline 3 stream CUDA:
   - H2D transfer chunk berikutnya.
   - Compute chunk aktif.
   - D2H transfer chunk sebelumnya.
3. Sinkronisasi via CUDA event agar overlap aman.

### C. Solusi Validasi dan Benchmark

1. CPU reference chunked untuk meniru perilaku chunk GPU.
2. CPU sequential global sebagai baseline serial murni.
3. Report mismatch + timing + throughput + speedup.

### D. Solusi Operasional/Profiling

1. Mode eksekusi terpisah: `parallel`, `sequential`, `both`.
2. Binary terpisah:
   - `dna_pipeline_parallel`
   - `dna_pipeline_sequential`
3. Folder output terpisah agar profiling tidak tercampur:
   - `prof/Parallel`
   - `prof/Sequential`
   - `prof/Both`

## 4. Apa yang Dilakukan Proyek Ini (Implementasi Nyata)

Secara end-to-end, program melakukan:

1. Membaca file input biner (`--input-file <path>`).
2. Mengonversi bit menjadi sekuens ACGT dengan mapping naif:
   - `00 -> A`
   - `01 -> C`
   - `10 -> G`
   - `11 -> T`
3. Menjalankan pipeline fitur DNA (GPU/CPU sesuai mode).
4. Menghitung:
   - GC violations
   - Homopolymer violations
   - Motif collisions
   - Entropy vector
5. Menjalankan validasi silang (jika mode `both`).
6. Menyimpan laporan ke file output sesuai mode.

## 5. Nilai/Manfaat yang Dihasilkan

1. Throughput jauh lebih tinggi pada mode paralel dibanding serial.
2. Jalur validasi membuat optimasi tetap dapat dipercaya.
3. Profiling lebih bersih karena workload parallel/sequential dipisah.
4. Input fleksibel: bisa dari file biner apa pun, bukan hanya data random.

## 6. Keterbatasan Saat Ini

1. Perilaku batas chunk untuk fitur windowed bisa berbeda dari model global kontinu.
2. Pemetaan file biner ke ACGT bersifat naif (langsung per 2-bit, tanpa model biologis tambahan).
3. Beberapa mismatch terhadap baseline global bisa muncul sebagai konsekuensi desain chunking.

## 7. Arah Pengembangan Lanjutan

1. Tambah halo overlap antar chunk untuk mengurangi mismatch batas.
2. Tambah mode input biologis (mis. FASTA/FASTQ parser).
3. Tambah export metrik per tahap ke format CSV/JSON untuk analisis otomatis.
4. Tambah NVTX marker agar timeline Nsight lebih mudah dibaca.

---

Dokumen ini ditujukan sebagai ringkasan konseptual proyek untuk kebutuhan laporan, presentasi, dan dokumentasi teknis tingkat menengah.
