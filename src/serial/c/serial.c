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

int main(void)
{
    struct Matrix source;
    struct FreqMatrix freq_domain;
    clock_t start, end;
    readMatrix(&source);
    freq_domain.size = source.size;

    start = clock();

    for (int k = 0; k < source.size; k++)
        for (int l = 0; l < source.size; l++)
            freq_domain.mat[k][l] = dft(&source, k, l);

    end = clock();

    double complex sum = 0.0;

    // print some of the matrix
    for (int k = 0; k < 3; k++)
    {
        printf("{");
        for (int l = 0; l < 3; l++)
        {
            double complex el = freq_domain.mat[k][l];
            printf("(%lf, %lf) ", creal(el), cimag(el));
        }
        printf("}\n");
    }

    // calculate the sum of the matrix
    for (int k = 0; k < source.size; k++)
    {   
        for (int l = 0; l < source.size; l++)
        {
            double complex el = freq_domain.mat[k][l];
            sum += el;
        }
    }

    sum /= source.size;
    printf("Average : (%lf, %lf)\n", creal(sum), cimag(sum));
    printf("Time: %f\n", ((double)(end - start)) / CLOCKS_PER_SEC);

    return 0;
}