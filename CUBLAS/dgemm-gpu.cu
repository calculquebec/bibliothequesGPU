// Convert this CPU code into GPU one. 
// Replace all CPU calls with the GPU ones and use CUBLAS library instead of MKL for matrix
// multiplication

// This example computes real matrix C=alpha*A*B+beta*C using Intel® MKL function dgemm, 
// where A, B, and  C are matrices and alpha and beta are double precision scalars

#define min(x,y) (((x) < (y)) ? (x) : (y))

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

int main()
{
    // Define the variables
    double *A, *B, *C;
    int m, n, k, i, j;
    double alpha, beta;

    // Set the size of the problem
    m = 2000, k = 2000, n = 2000;
    alpha = 1.0; beta = 0.0;

    // Allocate memory for the the matrices A, B, C
    printf (" Allocating memory for matrices aligned on 64-byte boundary for better \n"
            " performance \n\n");
    A = (double *)malloc( m*k*sizeof( double ) );
    B = (double *)malloc( k*n*sizeof( double ) );
    C = (double *)malloc( m*n*sizeof( double ) );

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

    // Perform matrix multiplication
    printf (" Computing matrix product using Intel® MKL dgemm function via CBLAS interface \n\n");
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, k, alpha, A, k, B, n, beta, C, n);
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

    // Free the memory of the matrices A, B, C
    printf ("\n Deallocating memory \n\n");
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    printf (" Example completed. \n\n");
    return 0;
}
