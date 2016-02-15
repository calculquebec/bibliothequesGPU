// Convert this CPU code to the GPU one. 
// Replace all the CPU calls with the GPU ones. Use MAGMA library instead of Intel MKL.

// This example solves a double precision system of linears equations A*X=B using MKL libraries


# define RAND_MAX 10000
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <mkl_lapack.h>

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
    

    // Solve the system A*X=B using Intel MKL libraries
    dgesv(&n, &nrhs, A, &lda, ipiv, B, &ldb, &info);

    // Free the memory
    free(A);
    free(ipiv);
    free(B);

}
