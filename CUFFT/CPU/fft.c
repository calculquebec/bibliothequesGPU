#define NX 256

#include <stdio.h>
#include <stdlib.h>
#include <fftw3.h>



int main(){

   /* Define FFT variables */
   fftw_complex *in, *out;
   fftw_plan plan;



   /* Allocate memory on CPU for FFT data */
   in = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * NX );
   out = (fftw_complex*) fftw_malloc ( sizeof ( fftw_complex ) * NX );


   /* Create FFT plan */
   plan = fftw_plan_dft_1d(NX, in, out, FFTW_FORWARD, FFTW_ESTIMATE);


   /* Perform complex-to-complex FFT transformation */
   fftw_execute(plan);

//   cufftExecC2C(plan, data, data, CUFFT_FORWARD);

   /* Destroy FFT plan */
   fftw_destroy_plan(plan);

   /* Free the memory */
   fftw_free(in);
   fftw_free(out);

}
