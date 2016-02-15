// This example solves a double precision system of linears equations A*X=B using MKL libraries


# define RAND_MAX 10000
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
//#include <mkl_lapack.h>
#include "magma.h"
#include "magma_lapack.h"
//#include <cuda_runtime_api.h>
//#include <cublas.h>

void fillRandomDouble(int m, int, double* a, double min, double max);

void fillRandomDouble(int m, int n, double* a, double min, double max)
{
    int i, j;

    srand(1);

    for (j=0; j<m; j++)
    {
        for (i=0; i<n; i++)
        {
            a[j*n+i] = min + (max-min) * rand()/RAND_MAX;
        }
    }
}


int main(void)
{
    int n=4096;
    int nrhs = n;
    int lda = n;
    int ldb = n;
    int info = 0;
    double gpu_time;

    // Define matrix variables
    double *A;
    int *ipiv;
    double *B;


    // Allocate matrices
    A = (double*) malloc(lda*n*sizeof(double));
    ipiv = (int*) malloc(n*sizeof(int));
    B = (double*) malloc(ldb*nrhs*sizeof(double));

    // Add some random data
    fillRandomDouble(lda, n, A, -10.0f, 10.0f);
    fillRandomDouble(ldb, nrhs, B, -10.0f, 10.0f);
   
    /* Initialize MAGMA */
    magma_init();

    /* Print available GPU devices */
    magma_print_devices();

 

    // Solve the system A*X=B using Intel MKL libraries
    gpu_time = magma_wtime();
    magma_dgesv( n, nrhs, A, lda, ipiv, B, ldb, &info );
    gpu_time = magma_wtime() - gpu_time;
    printf("info=%d %7.2f\n",info,gpu_time);
    // Free the memory
    free(A);
    free(ipiv);
    free(B);

}
