#define NX 256

#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>



int main(){

   /* Define FFT variables */
   cufftComplex *in, *out;
   cufftHandle plan;

   /* Set the GPU device */
   cudaSetDevice(0);

   /* Allocate memory on GPU for FFT data */
   cudaMalloc((void**)&in, NX*sizeof(cufftComplex));
   cudaMalloc((void**)&out, NX*sizeof(cufftComplex));


   /* Create FFT plan */
   cufftPlan1d(&plan, NX, CUFFT_C2C, 1);

   /* Perform complex-to-complex FFT transformation */
   cufftExecC2C(plan, in, out, CUFFT_FORWARD);
   printf("Checking error: %s\n",cudaGetErrorString(cudaGetLastError()));

   /* Destroy FFT plan */
   cufftDestroy(plan);

   /* Free the memory */
   cudaFree(in);
   cudaFree(out);

}
