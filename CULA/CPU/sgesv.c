// This example solves the linear equation A*X=B using Intel MKL libraries

# define RAND_MAX 10000
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <mkl_lapack.h>


void fillRandomFloat(int m, int, float* a, float min, float max);

void fillRandomFloat(int m, int n, float* a, float min, float max)
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
    int nrhs = 1;
    int n=10000;
    int lda = n;
    int ldb = n;
    int info = 0;

    float *A;
    int *ipiv;
    float *B;

    // Allocate memory 
    A = (float*) malloc(lda*n*sizeof(float));
    ipiv = (int*) malloc(n*sizeof(int));
    B = (float*) malloc(ldb*nrhs*sizeof(float));

    // Add some random data
    fillRandomFloat(lda, n, A, -10.0f, 10.0f);
    fillRandomFloat(ldb, nrhs, B, -10.0f, 10.0f);
    


    /* Perform matrix inversion with MKL on CPU */
    sgesv(&n, &nrhs, A, &lda, ipiv, B, &ldb, &info);



    // Free the memory
    free(A);
    free(ipiv);
    free(B);
}
