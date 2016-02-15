// This example computes real matrix C=alpha*A*B+beta*C using Intel® MKL function dgemm, 
// where A, B, and  C are matrices and alpha and beta are double precision scalars

#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

int main()
{
    // Define variables
    double *A, *B, *C;
    int m, n, k, i, j;
    double alpha, beta;

    double *d_A, *d_B, *d_C;

    // Set up the matrices
    m = 2000, k = 2000, n = 2000;
    printf (" Initializing data for matrix multiplication C=A*B for matrix \n"
            " A(%ix%i) and matrix B(%ix%i)\n\n", m, k, k, n);
    alpha = 1.0; beta = 0.0;

    // Allocate CPU memory
    printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
            " performance \n\n");
    A = (double *)malloc( m*k*sizeof( double ) );
    B = (double *)malloc( k*n*sizeof( double ) );
    C = (double *)malloc( m*n*sizeof( double ) );

    // Set GPU device
    cudaSetDevice(0);

    // Allocate GPU memory
    cudaMalloc((void**)&d_A, m*k*sizeof(double));
    cudaMalloc((void**)&d_B, k*n*sizeof(double));
    cudaMalloc((void**)&d_C, m*n*sizeof(double));

    // Initialize matrices
    printf (" Intializing matrix data \n\n");
    for (i = 0; i < (m*k); i++) {
        A[i] = (double)(i+1);
    }

    for (i = 0; i < (k*n); i++) {
        B[i] = (double)(-i-1);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
    }

    /* Copy data to GPU  */
    cublasSetVector(m*k, sizeof(double), A, 1, d_A, 1);
    cublasSetVector(k*n, sizeof(double), B, 1, d_B, 1);


    /* Initialize cuBLAS */
    cublasHandle_t handle;
    cublasCreate(&handle);

    /* Perform multiplication on GPU  */
    for(int i=0;i<20;i++) cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,m,n,k,&alpha, d_A, m, d_B, k, &beta, d_C, m); 


    /* Finalize cuBLAS  */
    cublasDestroy(handle);


    /* Copy results back to CPU */
    cublasGetVector(m*n, sizeof(double), d_C, 1, C, 1);

    /* Print the results */
    printf ("\n Computations completed.\n\n");
    printf (" Top left corner of matrix A: \n");
    for (i=0; i<min(m,6); i++) {
      for (j=0; j<min(k,6); j++) {
        printf ("%12.0f", A[j+i*k]);
      }
      printf ("\n");
    }

    printf ("\n Top left corner of matrix B: \n");
    for (i=0; i<min(k,6); i++) {
      for (j=0; j<min(n,6); j++) {
        printf ("%12.0f", B[j+i*n]);
      }
      printf ("\n");
    }
    
    printf ("\n Top left corner of matrix C: \n");
    for (i=0; i<min(m,6); i++) {
      for (j=0; j<min(n,6); j++) {
        printf ("%12.5G", C[j+i*n]);
      }
      printf ("\n");
    }

    // Free CPU memory
    printf ("\n Deallocating memory \n\n");
    free(A);
    free(B);
    free(C);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf (" Example completed. \n\n");
    return 0;
}
