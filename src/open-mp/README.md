# Implementasi Paralelisasi dengan OpenMP

## Cara Kerja Paralelisasi Program
Implementasi paralelisasi program 2D DFT dilakukan sebagai 
berikut.
1. Matriks masukan serta matriks frekuensi hasil perhitungan
   diinisialisasi oleh fungsi `main`.
2. Fungsi `readMatrix` akan dipanggil oleh fungsi `main` untuk membaca matriks
   masukan dari file berekstensi `.txt` pada folder `test_case`.
3. Matriks frekuensi kemudian diisi menggunakan hasil pemanggilan fungsi `dft` yang
   dilakukan menggunakan for loop yang ditandai dengan `pragma omp parallel for`.
4. Setiap proses yang menjalankan fungsi `dft` yang dipanggil fungsi `main` akan 
   melakukan perhitungan frekuensi menggunakan algoritma 2D DFT dan menyimpan
   hasilnya di variabel lokal `local_element` yang dimiliki masing-masing proses.
5. Hasil perhitungan ini dijumlahkan ke variabel bersama `element`; proses 
   penjumlahan ini ditandai sebagai critical section menggunakan `pragma omp 
   critical`.
6. Matriks frekuensi yang telah diisi kemundian ditampilkan ke layar bersama dengan 
   rata-rata elemen matriks frekuensi.

## Cara Program Membagi Data
Pembagian data dilakukan ketika compiler menemukan penanda `pragma omp parallel 
for`. Compiler akan membuat sejumlah threads untuk memenuhi directive `parallel`; 
banyak thread yang dibuat bergantung pada definisi implementasi OpenMP. Setelah 
itu, compiler akan mendistribusikan iterasi pada program untuk memenuhi clause 
`for`.
