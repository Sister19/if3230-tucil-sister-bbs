# Implementasi Paralelisasi dengan CUDA

## Cara Kerja Paralelisasi Program
Implementasi paralelisasi program 2D DFT dilakukan sebagai
berikut.
1. Matriks masukan serta matriks frekuensi hasil perhitungan milik perangkat
   diinisialisasi oleh fungsi `main`.
2. Fungsi `readMatrix` akan dipanggil oleh fungsi `main` untuk membaca matriks 
   masukan dari file berekstensi `.txt` pada folder `test_case`.
3. Banyak thread per block (`threadsPerBlock`) dan banyak block per grid 
   (`blocksPerGrid`) diinisiasi oleh fungsi `main`.
4. Matriks masukan serta matriks frekuensi hasil perhitungan milik GPU
   diinisialisasi serta dialokasikan memorinya oleh fungsi `main`.
5. Matriks masukan dikirim dari memori perangkat menuju memori GPU menggunakan 
   `cudaMemcpyHostToDevice`.
6. Komputasi DFT dilakukan menggunakan fungsi `computeDFT` yang dipanggil oleh 
   fungsi `main` untuk mengeksekusi kernel.
7. Matriks hasil komputasi DFT dikirim dari memori GPU menuju memori perangkat 
   menggunakan `cudaMemcpyDeviceToHost`.
8. Memori GPU yang sebelumnya telah dialokasikan untuk perhitungan DFT dibebaskan 
   menggunakan `cudaFree`.
9. Untuk memastikan waktu komputasi dihitung dengan akurat, digunakan 
   `cudaDeviceSynchronize` untuk memastikan semua thread telah selesai melakukan 
   perhitungan sebelum waktu eksekusi dihitung.

## Cara Program Membagi Data
Pembagian data dilakukan saat pemanggilan fungsi `computeDFT`. Pembagian ini 
dilakukan secara merata pada sebuah `grid` yang terdiri atas `source.size / 32 
blocks` yang terdiri atas `32 threads`. Setiap `thread` akan mendapatkan banyak 
elemen matriks masukan yang sama.
