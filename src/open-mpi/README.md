# Implementasi Paralelisasi dengan Open MPI

## Cara kerja Paralelisasi program

Dasar dari algoritma 2D DFT dan implementasi pada file [`serial.c`](../serial/c/serial.c) adalah melakukan perhitungan pada setiap elemen matriks dan menggabungkan hasilnya pada suatu variabel reducer. Penjalasan implementasinya adalah sebagai berikut.

1. Inisiasi dua matriks, matriks masukan serta matriks frekuensi hasil perhitungan.
2. Fungsi utama `main` menginisiasi MPI (`MPI_Init`), mendapatkan jumlah proses (`MPI_Comm_size`) dan mendapatkan rank proses (`MPI_Comm_rank`). Rank proses adalah nomor proses yang dijalankan, dimulai dari 0.
3. Setelah itu, proses dengan rank 0 akan memanggil fungsi `read_matrix` untuk membaca matriks awal dari file masukan berekstensi `.txt` pada folder [test_case](../../test_case/) 
4. Fungsi `broadcast_matrix` dipanggil untuk membagikan seluruh elemen matriks (Penjelasan pembagian data ada di bagian bawah). Fungsi ini memanggil fungsi `MPI_Bcast` untuk membagikan data. Selain itu, fungsi ini juga menghitung batas bawah dan batas bawah perhitungan masing-masing proses.
5. Setelah itu, fungsi `compute_freq_domain` dipanggil oleh masing-masing proses untuk melakukan perhitungan pada sebagian elemen matriks yang telah dibagikan. Fungsi ini memanggil fungsi formula `dft` yang merupakan implementasi dari algoritma 2D DFT.
6. selanjutnya, fungsi `gather_freq_domain` dipanggil untuk menggabungkan hasil perhitungan dari masing-masing proses menjadi matriks frekuensi utuh. Fungsi ini memanggil fungsi `MPI_Gather` untuk menggabungkan data.
7. Setelah itu, proses dengan *rank* 0 akan memanggil fungsi `print_result` yang akan menuliskan matriks ke layar serta menghitung rata-rata dari total keseluruhan elemen matriks frekuensi.
8. Terakhir, fungsi `MPI_Finalize` dipanggil untuk menutup proses MPI.

## Cara program membagikan data antar-proses

Pembagian data dilakukan saat pemanggilan fungsi `broadcast_matrix`. Fungsi `MPI_Bcast` dipanggil agar setiap proses menerima matriks masukan utuh. Matriks dibagi berdasarkan jumlah proses yang dijalankan. Jumlah baris yang dibagi adalah jumlah baris matriks dibagi jumlah proses. Pembagian matriks dilakukan berdasarkan jumlah baris, dengan memprioritaskan kerataan persebaran dan mendahulukan proses dengan rank yang lebih rendah terlebih dahulu. Dihitung variabel `local_rows_start` dan local_rows_end` sedemikian sehingga setiap proses hanya akan memproses matriks dari baris ke-`local_rows_start` sampai baris ke-`local_rows_end - 1`. Tidak lupa `MPI_Bcast` kembali dipanggil untuk membagikan banyaknya baris yang dihitung oleh proses utama.
