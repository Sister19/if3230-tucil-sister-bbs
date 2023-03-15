#include <cuComplex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_N 512

struct Matrix
{
  int size;
  double mat[MAX_N][MAX_N];
};

struct FreqMatrix
{
  int size;
  cuDoubleComplex mat[MAX_N][MAX_N];
};

void readMatrix(struct Matrix *m)
{
  scanf("%d", &(m->size));
  for (int i = 0; i < m->size; i++)
    for (int j = 0; j < m->size; j++)
      scanf("%lf", &(m->mat[i][j]));
}

// cuDoubleComplex dft(struct Matrix *mat, int k, int l)
// {
//   double complex element = 0.0;
//   for (int m = 0; m < mat->size; m++)
//   {
//     for (int n = 0; n < mat->size; n++)
//     {
//       double complex arg = (k * m / (double)mat->size) + (l * n / (double)mat->size);
//       double complex exponent = cexp(-2.0I * M_PI * arg);
//       element += mat->mat[m][n] * exponent;
//     }
//   }
//   return element / (double)(mat->size * mat->size);
// }

__global__ void computeDFT(struct Matrix *src, struct FreqMatrix *dest, int k, int l)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < src->size && j < src->size)
  {
    cuDoubleComplex element = make_cuDoubleComplex(0.0, 0.0);
    for (int m = 0; m < src->size; m++)
    {
      for (int n = 0; n < src->size; n++)
      {
        // double complex arg = (k * m / (double)src->size) + (l * n / (double)src->size);
        // double complex exponent = cexp(-2.0I * M_PI * arg);
        cuDoubleComplex exponent = make_cuDoubleComplex(0.0, -2.0 * M_PI * (k * m / (double)src->size) + (l * n / (double)src->size));
        element = cuCadd(element, cuCmul(make_cuDoubleComplex(src->mat[m][n], 0.0), exponent));
      }
    }
    dest->mat[i][j] = make_cuDoubleComplex(0.0, 0.0);
    dest->mat[i][j] = cuCdiv(element, make_cuDoubleComplex(src->size * src->size, 0.0));
  }
}

int main(void)
{
  struct Matrix source;
  struct FreqMatrix freq_domain;
  clock_t start, end;
  readMatrix(&source);
  freq_domain.size = source.size;

  start = clock();
  int threads = 32;
  dim3 threadsPerBlock(threads, threads);
  dim3 blocksPerGrid(source.size / threads, source.size / threads);

  for (int k = 0; k < source.size; k++)
  {
    for (int l = 0; l < source.size; l++)
    {
      struct Matrix *dev_source;
      struct FreqMatrix *dev_dest;
      cudaMalloc((void **)&dev_source, sizeof(struct Matrix));
      cudaMalloc((void **)&dev_dest, sizeof(struct FreqMatrix));
      cudaMemcpy(dev_source, &source, sizeof(struct Matrix), cudaMemcpyHostToDevice);
      computeDFT<<<blocksPerGrid, threadsPerBlock>>>(dev_source, dev_dest, k, l);
      freq_domain.mat[k][l] = make_cuDoubleComplex(0.0, 0.0);
      cudaMemcpy(&freq_domain.mat[k][l], &dev_dest->mat[k][l], sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
      cudaFree(dev_source);
      cudaFree(dev_dest);
    }
  }
  cudaDeviceSynchronize();
  end = clock();

  // double complex sum = 0.0
  cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
  for (int k = 0; k < source.size; k++)
  {
    for (int l = 0; l < source.size; l++)
    {
      // double complex el = freq_domain.mat[k][l];
      // printf("(%lf, %lf) ", creal(el), cimag(el));
      // sum += el;
      sum = cuCadd(sum, freq_domain.mat[k][l]);
      printf("(%lf, %lf) ", cuCreal(freq_domain.mat[k][l]), cuCimag(freq_domain.mat[k][l]));
    }
    printf("\n");
  }
  // make_cuDoubleComplex(sum, 0,0) /= source.size;
  printf("Average : (%lf, %lf)\n", cuCreal(sum), cuCimag(sum));
  printf("Time: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

  return 0;
}