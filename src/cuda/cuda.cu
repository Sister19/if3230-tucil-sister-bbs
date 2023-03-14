#include <complex.h>
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
  double complex mat[MAX_N][MAX_N];
};

void readMatrix(struct Matrix *m)
{
  scanf("%d", &(m->size));
  for (int i = 0; i < m->size; i++)
    for (int j = 0; j < m->size; j++)
      scanf("%lf", &(m->mat[i][j]));
}

double complex dft(struct Matrix *mat, int k, int l)
{
  double complex element = 0.0;
  for (int m = 0; m < mat->size; m++)
  {
    for (int n = 0; n < mat->size; n++)
    {
      double complex arg = (k * m / (double)mat->size) + (l * n / (double)mat->size);
      double complex exponent = cexp(-2.0I * M_PI * arg);
      element += mat->mat[m][n] * exponent;
    }
  }
  return element / (double)(mat->size * mat->size);
}

__global void computeMatrix(struct Matrix *src, struct Matrix *dest, int k, int l)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < src->size && j < src->size)
  {
    double complex element = 0.0;
    for (int m = 0; m < src->size; m++)
    {
      for (int n = 0; n < src->size; n++)
      {
        double complex arg = (k * m / (double)src->size) + (l * n / (double)src->size);
        double complex exponent = cexp(-2.0I * M_PI * arg);
        element += src->mat[m][n] * exponent;
      }
    }
    dest->mat[i][j] = element / (double)(src->size * src->size);
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
  int threadsPerBlock = 32;
  dim3 threadsPerBlock(threadsPerBlock, threadsPerBlock);
  dim3 blocksPerGrid(source.size / threadsPerBlock, source.size / threadsPerBlock);

  for (int k = 0; k < source.size; k++)
  {
    for (int l = 0; l < source.size; l++)
    {
      struct Matrix *dev_source, *dev_dest;
      cudaMalloc((void **)&dev_source, sizeof(struct Matrix));
      cudaMalloc((void **)&dev_dest, sizeof(struct Matrix));
      cudaMemcpy(dev_source, &source, sizeof(struct Matrix), cudaMemcpyHostToDevice);
      computeMatrix<<<blocksPerGrid, threadsPerBlock>>>(dev_source, dev_dest, k, l);
      cudaMemcpy(&freq_domain.mat[k][l], &dev_dest->mat[k][l], sizeof(double complex), cudaMemcpyDeviceToHost);
      cudaFree(dev_source);
      cudaFree(dev_dest);
    }
  }
  cudaDeviceSynchronize();
  end = clock();

  double complex sum = 0.0;
  for (int k = 0; k < source.size; k++)
  {
    for (int l = 0; l < source.size; l++)
    {
      double complex el = freq_domain.mat[k][l];
      printf("(%lf, %lf) ", creal(el), cimag(el));
      sum += el;
    }
    printf("\n");
  }
  sum /= source.size;
  printf("Average : (%lf, %lf)\n", creal(sum), cimag(sum));
  printf("Time: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

  return 0;
}