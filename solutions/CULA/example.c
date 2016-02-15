#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cula_lapack.h>

int main(void)
{
    int M = 8192;
    int N = M;
    
    culaStatus status;

    float* A = NULL;
    float* B = NULL;

    A = (float*)malloc(M*N*sizeof(float));
    B = (float*)malloc(N*sizeof(float));
    if(!A || !B) exit(EXIT_FAILURE);
    memset(A, 0, M*N*sizeof(float));
    memset(B, 0, N*sizeof(float));

    status = culaInitialize();

    status = culaSgeqrf(M, N, A, M, B);
    
    culaShutdown();

    free(A);
    free(B);

    return EXIT_SUCCESS;
}
