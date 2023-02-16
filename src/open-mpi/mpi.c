// mpicc mpi.c -o mpi

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_N 512

struct Matrix
{
    int size;
    double mat[MAX_N * MAX_N];
};

struct FreqMatrix
{
    int size;
    double complex mat[MAX_N * MAX_N];
};

struct Matrix source;
struct FreqMatrix freq_domain;
int local_rows_start;
int local_rows_end;
int local_size;
int world_size;
int world_rank;

void readMatrix(struct Matrix *m)
{
    scanf("%d", &(m->size));
    for (int i = 0; i < m->size; i++)
        for (int j = 0; j < m->size; j++)
            scanf("%lf", &(m->mat[i * m->size + j]));
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
            element += mat->mat[m * mat->size + n] * exponent;
        }
    }
    return element / (double)(mat->size * mat->size);
}

void init_matrix()
{
    readMatrix(&source);
}

void broadcast_matrix()
{
    MPI_Bcast(&source, sizeof(source), MPI_BYTE, 0, MPI_COMM_WORLD);
    freq_domain.size = source.size;
    if (world_rank != world_size - 1)
    {
        local_rows_start = world_rank * (source.size / world_size);
        local_rows_end = (world_rank + 1) * (source.size / world_size);
    }
    else
    {
        local_rows_start = world_rank * (source.size / world_size) + (source.size % world_size);
        local_rows_end = source.size;
    }
    local_size = local_rows_end - local_rows_start;
}

void compute_freq_domain()
{
    for (int k = local_rows_start; k < local_rows_end; k++)
        for (int l = 0; l < source.size; l++)
            freq_domain.mat[k * freq_domain.size + l] = dft(&source, k, l);
}

void gather_freq_domain()
{
    printf("\n%d %p %p %p\n", world_rank, &freq_domain, &freq_domain + sizeof(double complex) * (freq_domain.size + local_rows_start), sizeof(double complex) * (local_rows_end - local_rows_start + 1));
    MPI_Gather(
        &(freq_domain.mat[local_rows_start * freq_domain.size]),
        local_size * freq_domain.size,
        MPI_C_DOUBLE_COMPLEX,
        &(freq_domain.mat),
        local_size * freq_domain.size,
        MPI_C_DOUBLE_COMPLEX,
        0,
        MPI_COMM_WORLD);
}

int main(void)
{
    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0)
        init_matrix();

    broadcast_matrix();

    compute_freq_domain();

    gather_freq_domain();

    if (world_rank == 0)
    {
        double complex sum = 0.0;
        for (int k = 0; k < source.size; k++)
        {
            for (int l = 0; l < source.size; l++)
            {
                double complex el = freq_domain.mat[k * freq_domain.size + l];
                printf("(%lf, %lf) ", creal(el), cimag(el));
                sum += el;
            }
            printf("\n");
        }

        sum /= source.size;
        printf("Average : (%lf, %lf)", creal(sum), cimag(sum));
    }


        MPI_Finalize();

    return 0;
}