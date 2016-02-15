/*
 * Copyright (C) 2009-2012 EM Photonics, Inc.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to EM Photonics ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code may
 * not redistribute this code without the express written consent of EM
 * Photonics, Inc.
 *
 * EM PHOTONICS MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED
 * WARRANTY OF ANY KIND.  EM PHOTONICS DISCLAIMS ALL WARRANTIES WITH REGARD TO
 * THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.  IN NO EVENT SHALL EM
 * PHOTONICS BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
 * DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
 * PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
 * SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as that
 * term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of "commercial
 * computer  software"  and "commercial computer software documentation" as
 * such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the
 * U.S. Government only as a commercial end item.  Consistent with 48
 * C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the source code with only those rights set
 * forth herein. 
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code, the
 * above Disclaimer and U.S. Government End Users Notice.
 *
 */

/*
 * CULA Example: BENCHMARK
 *
 * This example benchmarks CULA against Intel MKL.  For each function, this
 * program sweeps through a range of sizes.  You may customize the start, stop,
 * and step size by specifying each of these as arguments to this program.
 *
 *    uage:   benchmark start stop step
 *    e.g.    benchmark 1024 4096 256
 *
 * Note: This example requires Intel MKL.  If you do not have MKL, you may use
 * the precompiled binary 'benchmark_'.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

// Timers
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/time.h>
#endif

#include <cula_lapack.h>

// Intel MKL
#include <mkl_lapack.h>
#ifdef _MSC_VER
#   ifdef _WIN64
#       pragma comment(lib, "mkl_intel_lp64.lib")
#   else
#       pragma comment(lib, "mkl_intel_c.lib")
#   endif
#   pragma comment(lib, "libiomp5md.lib")
#   pragma comment(lib, "mkl_intel_thread.lib")
#   pragma comment(lib, "mkl_core.lib")
#endif

// Declarations
int imax(int a, int b);
int imin(int a, int b);
double getHighResolutionTime(void);
void fillRandomSingle(int m, int, float* a, float min, float max);
void fillRandomDouble(int m, int, double* a, double min, double max);
int strcmp_ci(const char* left, const char* right);
void initializeMkl();

void printUsage();
void printHeader(const char* title);
void printProblemSize(int n);
void printRuntime(double t);
void printSpeedup(double culaTime, double mklTime);
void printCulaError(culaStatus status);

culaStatus benchSgeqrf(int n);
culaStatus benchSgetrf(int n);
culaStatus benchSgels(int n);
culaStatus benchSgglse(int n);
culaStatus benchSgesvd(int n);
culaStatus benchSgesv(int n);

#ifdef CULA_PREMIUM
culaStatus benchSsyev(int n);
culaStatus benchDgeqrf(int n);
culaStatus benchDgetrf(int n);
culaStatus benchDgels(int n);
culaStatus benchDgglse(int n);
culaStatus benchDgesvd(int n);
culaStatus benchDgesv(int n);
culaStatus benchDsyev(int n);
#endif

typedef culaStatus (*BenchmarkFunctionPointer)(int);

typedef struct
{
    const char* name;
    int run;
    BenchmarkFunctionPointer func;
}BenchmarkFunction;


// Main
int main(int argc, char** argv)
{
    int start = 4096;
    int stop = 8192;
    int step = 1024;

    BenchmarkFunction functions[] =
    {
        { "SGEQRF", 0, benchSgeqrf },
        { "SGETRF", 0, benchSgetrf },
        { "SGELS",  0, benchSgels  },
        { "SGGLSE", 0, benchSgglse },
        { "SGESV",  0, benchSgesv  },
        { "SGESVD", 0, benchSgesvd },
#ifdef CULA_PREMIUM
        { "SSYEV",  0, benchSsyev  },
        { "DGEQRF", 0, benchDgeqrf },
        { "DGETRF", 0, benchDgetrf },
        { "DGELS",  0, benchDgels  },
        { "DGGLSE", 0, benchDgglse },
        { "DGESV",  0, benchDgesv  },
        { "DGESVD", 0, benchDgesvd },
        { "DSYEV",  0, benchDsyev  },
#endif
    };

    int numFunctions = sizeof(functions)/sizeof(BenchmarkFunction);
    int numSpecified = 0;
    int numToRun = 0;

    int i,j;
    int n;
    char* arg;
    char group = 'N';
    int negate = 0;
    int rangeArg = 0;
    int validArg = 0;
    culaStatus status;

    for(i = 1; i < argc; ++i)
    {
        arg = argv[i];
        validArg = 0;
        negate = 0;

        if(strcmp_ci(arg, "-h") == 0
        || strcmp_ci(arg, "--help") == 0
        || strcmp_ci(arg, "help") == 0
        || strcmp_ci(arg, "/?") == 0)
        {
            printUsage();
            return EXIT_SUCCESS;
        }

        if(arg[0] == '-')
        {
            ++arg;
            negate = 1;
        }
        else if(arg[0] == '+')
            ++arg;

        n = atoi(arg);
        if(n != 0)
        {
            if(negate)
            {
                printf("Invalid start, stop, or step size (%s)\n", arg);
                printUsage();
                return EXIT_FAILURE;
            }

            if(rangeArg == 0)
                start = n;
            else if(rangeArg == 1)
                stop = n;
            else if(rangeArg == 2)
                step = n;
            else
            {
                printf("Invalid start, stop, or step size (%s)\n", arg);
                printUsage();
                return EXIT_FAILURE;
            }
            ++rangeArg;
            continue;
        }

        for(j = 0; j < numFunctions; ++j)
        {
            if(strcmp_ci(arg, functions[j].name) == 0)
            {
                if(negate)
                    functions[j].run = -1;
                else
                {
                    functions[j].run = 1;
                    ++numSpecified;
                }

                validArg = 1;
                break;
            }
        }

        if(strcmp_ci(arg, "single") == 0)
            group = 'S';
        else if(strcmp_ci(arg, "double") == 0)
            group = 'D';
        if(group != 'N')
        {
            for(j = 0; j < numFunctions; ++j)
            {
                if(functions[j].name[0] == group)
                {
                    if(negate)
                        functions[j].run = -1;
                    else
                    {
                        functions[j].run = 1;
                        ++numSpecified;
                    }
                }
            }
            group = 'N';
            continue;
        }

        if(!validArg)
        {
            printf("\nInvalid Argument: %s\n", argv[i]);
            printUsage();
            return EXIT_FAILURE;
        }
    }

    if(start < 0  || stop < 0 || start > stop || step <= 0)
    {
        printf("Invalid start, stop, or step size\n");
        printUsage();
        return EXIT_FAILURE;
    }

    // If none are specified explicitly, run them all
    if(numSpecified == 0)
    {
        for(j = 0; j < numFunctions; ++j)
        {
            if(functions[j].run == -1)
                functions[j].run = 0;
            else
                functions[j].run = 1;
        }
    }
    
    // Count the number of functions to run
    for(j = 0; j < numFunctions; ++j)
    {
        numToRun += functions[j].run;
    }

    // If there are no functions to run, exit
    if(numToRun == 0)
    {
        printf("Error: no functions to run.\n");
        return EXIT_FAILURE;
    }

    // Initialize CULA
    printf("Initializing CULA...\n");
    status = culaInitialize();

    // Early exit if CULA fails to initialize (no GPU, etc)
    if(status)
    {
        printf("CULA failed to initialized with error message:%s.\n", culaGetStatusString(status));
        return EXIT_FAILURE;
    }

    // Initialize MKL
    printf("Initializing MKL...\n");
    initializeMkl();

    printf("\n");
    printf("Benchmarking the following functions:\n");
    printf("-------------------------------------\n");
    for(i = 0; i < numFunctions; ++i)
    {
        if(functions[i].run)
        {
            printf("             %s\n", functions[i].name);
        }
    }
    printf("-------------------------------------\n");
    printf("\n");

    // Benchmarks
    for(i = 0; i < numFunctions; ++i)
    {
        if(functions[i].run)
        {
            printHeader(functions[i].name);
            for (n=start; n<=stop; n+=step)
            {
                status = functions[i].func(n);
                if(status != culaNoError)
                    break;
            }
        }
    }

    // Shutdown
    culaShutdown();

    return EXIT_SUCCESS;
}

// Cross platform high resolution timer
#ifdef _WIN32
double getHighResolutionTime(void)
{
    double freq;
    double seconds;
    LARGE_INTEGER end_time;
    LARGE_INTEGER performance_frequency_hz;

    QueryPerformanceCounter(&end_time);
    QueryPerformanceFrequency(&performance_frequency_hz);

    seconds = (double) end_time.QuadPart;
    freq = (double) performance_frequency_hz.QuadPart;
    seconds /= freq;

    return seconds;
}
#else
double getHighResolutionTime(void)
{
    struct timeval tod;

    gettimeofday(&tod, NULL);
    double time_seconds = (double) tod.tv_sec + ((double) tod.tv_usec / 1000000.0);
    return time_seconds;
}
#endif

void printUsage()
{
    printf("\n");
    printf("usage: [functions to run] [start stop step]\n");
    printf("\n");
    printf("    As string arguments are given, these are compared against function names\n"
           "        to specify which functions to run.  Functions may be prepended with\n"
           "        - to turn them off.  If no functions are explicitly specified, \n"
           "        all functions will be run.\n");
    printf("        \n");
    printf("    Data classes can also be specified.  Use -single or -double to turn off\n"
           "        single or double-precision functions, respectively\n");
    printf("        \n");
    printf("    As numeric arguments are given, start, stop, and step are filled in\n"
           "        this respective order\n");
    printf("\n");
    printf("    e.g. benchmark\n"
           "         runs all for sizes of 4096 to 8192 stepping by 1024 (default)\n");
    printf("    e.g. benchmark sgetrf sgesv\n"
           "         runs sgetrf and sgesv only\n");
    printf("    e.g. benchmark -sgesvd\n"
           "         runs all but sgesvd\n");
    printf("    e.g. benchmark sgeqrf 2048 4096 256\n"
           "         runs sgeqrf for sizes of 2048 to 4096 stepping by 256\n");
}

void printHeader(const char* name)
{
    printf("\n     -- %s Benchmark  --\n\n", name);
    printf(" Size   CULA (s)    MKL (s)   Speedup\n");
    printf("------ ---------- ---------- ---------\n");
    fflush(stdout);
}

void printProblemSize(int n)
{
    printf("%5.0d", n);
    fflush(stdout);
}

void printRuntime(double t)
{
    printf("%11.2f", t);
    fflush(stdout);
}

void printSpeedup(double cula_time, double mkl_time)
{
    printf("%10.4f\n",  mkl_time/cula_time);
    fflush(stdout);
}


// Min / Max
int imax(int a, int b)
{
    return (((a) > (b)) ? (a) : (b));
}

int imin(int a, int b)
{
    return (((a) < (b)) ? (a) : (b));
}

//
// Quasi-random data generater
//

void fillRandomSingle(int m, int n, float* a, float min, float max)
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

int strcmp_ci(const char* left, const char* right)
{
#if defined(_MSC_VER)
    return _stricmp (left, right);
#elif defined(__CYGWIN__) || defined(__MINGW32__)
    return stricmp (left, right);
#elif defined(__linux) || __APPLE__ & __MACH__
    return strcasecmp (left, right);
#else
#error "Unsupported compiler"
    return -1;
#endif
}

//
// Simple benchmarks
//

culaStatus benchSgeqrf(int n)
{
    int m = n;
    int lda = m;
    int k = imin(m,n);
    int info = 0;
    int lwork = -1;

    float* a_cula = NULL;
    float* a_mkl = NULL;
    float* tau_cula = NULL;
    float* tau_mkl = NULL;
    float* work_mkl = NULL;

    double start_time, end_time;
    double cula_time, mkl_time;
    culaStatus status = culaNoError;

    printProblemSize(n);

    a_cula = (float*) malloc(lda*n*sizeof(float));
    a_mkl = (float*) malloc(lda*n*sizeof(float));
    tau_cula = (float*) malloc(k*sizeof(float));
    tau_mkl = (float*) malloc(k*sizeof(float));
    work_mkl = (float*) malloc(1*sizeof(float));

    if(!a_cula || !a_mkl || !tau_cula || !tau_mkl || !work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchSgeqrf;
    }

    // Add some random data
    fillRandomSingle(lda, n, a_cula, -10.0f, 10.0f);
    memcpy(a_mkl, a_cula, lda*n*sizeof(float));

    // Run CULA's version
    start_time = getHighResolutionTime();
    status = culaSgeqrf(m, n, a_cula, lda, tau_cula);
    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    // Check CULA errors
    if(status)
    {
        printCulaError(status);
        goto endBenchSgeqrf;
    }

    printRuntime(cula_time);

    // Prepare MKL benchmark
    sgeqrf(&m, &n, a_mkl, &lda, tau_mkl, work_mkl, &lwork, &info);   // Worksize query
    lwork = (int) work_mkl[0];
    free(work_mkl);
    work_mkl = (float*) malloc(lwork*sizeof(float));

    if(!work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchSgeqrf;
    }

    // Run MKL's version
    start_time = getHighResolutionTime();
    sgeqrf(&m, &n, a_mkl, &lda, tau_mkl, work_mkl, &lwork, &info);
    end_time = getHighResolutionTime();
    mkl_time = end_time - start_time;

    if(info != 0)
    {
        printf(" Intel MKL Error: (%d)\n", info);
        goto endBenchSgeqrf;
    }

    printRuntime(mkl_time);
    printSpeedup(cula_time, mkl_time);

endBenchSgeqrf:
    free(a_cula);
    free(tau_cula);
    free(a_mkl);
    free(tau_mkl);
    free(work_mkl);
    return status;
}

#ifdef CULA_PREMIUM
culaStatus benchDgeqrf(int n)
{
    int m = n;
    int lda = m;
    int k = imin(m,n);
    int info = 0;
    int lwork = -1;

    double* a_cula = NULL;
    double* a_mkl = NULL;
    double* tau_cula = NULL;
    double* tau_mkl = NULL;
    double* work_mkl = NULL;

    double start_time, end_time;
    double cula_time, mkl_time;
    culaStatus status = culaNoError;

    printProblemSize(n);

    a_cula = (double*) malloc(lda*n*sizeof(double));
    a_mkl = (double*) malloc(lda*n*sizeof(double));
    tau_cula = (double*) malloc(k*sizeof(double));
    tau_mkl = (double*) malloc(k*sizeof(double));
    work_mkl = (double*) malloc(1*sizeof(double));

    if(!a_cula || !a_mkl || !tau_cula || !tau_mkl || !work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchDgeqrf;
    }

    // Add some random data
    fillRandomDouble(lda, n, a_cula, -10.0f, 10.0f);
    memcpy(a_mkl, a_cula, lda*n*sizeof(double));

    // Run CULA's version
    start_time = getHighResolutionTime();
    status = culaDgeqrf(m, n, a_cula, lda, tau_cula);
    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    // Check CULA errors
    if(status)
    {
        printCulaError(status);
        goto endBenchDgeqrf;
    }

    printRuntime(cula_time);

    // Prepare MKL benchmark
    dgeqrf(&m, &n, a_mkl, &lda, tau_mkl, work_mkl, &lwork, &info);   // Worksize query
    lwork = (int) work_mkl[0];
    free(work_mkl);
    work_mkl = (double*) malloc(lwork*sizeof(double));

    if(!work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchDgeqrf;
    }

    // Run MKL's version
    start_time = getHighResolutionTime();
    dgeqrf(&m, &n, a_mkl, &lda, tau_mkl, work_mkl, &lwork, &info);
    end_time = getHighResolutionTime();
    mkl_time = end_time - start_time;

    if(info != 0)
    {
        printf(" Intel MKL Error: (%d)\n", info);
        goto endBenchDgeqrf;
    }

    printRuntime(mkl_time);
    printSpeedup(cula_time, mkl_time);

endBenchDgeqrf:
    free(a_cula);
    free(tau_cula);
    free(a_mkl);
    free(tau_mkl);
    free(work_mkl);
    return status;
}
#endif

culaStatus benchSgetrf(int n)
{
    int m = n;
    int lda = m;
    int k = imin(m,n);
    int info = 0;

    float* a_cula = NULL;
    float* a_mkl = NULL;
    int* ipiv_cula = NULL;
    int* ipiv_mkl = NULL;

    double start_time, end_time;
    double cula_time, mkl_time;
    culaStatus status = culaNoError;

    printProblemSize(n);

    a_cula = (float*) malloc(lda*n*sizeof(float));
    a_mkl = (float*) malloc(lda*n*sizeof(float));
    ipiv_cula = (int*) malloc(k*sizeof(int));
    ipiv_mkl = (int*) malloc(k*sizeof(int));

    if(!a_cula || !a_mkl || !ipiv_cula || !ipiv_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchSgetrf;
    }

    // Add some random data
    fillRandomSingle(lda, n, a_cula, 1.0f, 2.0f);
    memcpy(a_mkl, a_cula, lda*n*sizeof(float));

    // Run CULA's version
    start_time = getHighResolutionTime();
    status = culaSgetrf(m, n, a_cula, lda, ipiv_cula);
    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    // Check CULA errors
    if(status)
    {
        printCulaError(status);
        goto endBenchSgetrf;
    }

    printRuntime(cula_time);

    // Run MKL's version
    start_time = getHighResolutionTime();
    sgetrf(&m, &n, a_mkl, &lda, ipiv_mkl, &info);
    end_time = getHighResolutionTime();
    mkl_time = end_time - start_time;

    if(info != 0)
    {
        printf(" Intel MKL Error: (%d)\n", info);
        goto endBenchSgetrf;
    }

    printRuntime(mkl_time);
    printSpeedup(cula_time, mkl_time);

endBenchSgetrf:
    free(a_cula);
    free(ipiv_cula);
    free(a_mkl);
    free(ipiv_mkl);
    return status;
}

#ifdef CULA_PREMIUM
culaStatus benchDgetrf(int n)
{
    int m = n;
    int lda = m;
    int k = imin(m,n);
    int info = 0;

    double* a_cula = NULL;
    double* a_mkl = NULL;
    int* ipiv_cula = NULL;
    int* ipiv_mkl = NULL;

    double start_time, end_time;
    double cula_time, mkl_time;
    culaStatus status = culaNoError;

    printProblemSize(n);

    a_cula = (double*) malloc(lda*n*sizeof(double));
    a_mkl = (double*) malloc(lda*n*sizeof(double));
    ipiv_cula = (int*) malloc(k*sizeof(int));
    ipiv_mkl = (int*) malloc(k*sizeof(int));

    if(!a_cula || !a_mkl || !ipiv_cula || !ipiv_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchDgetrf;
    }

    // Add some random data
    fillRandomDouble(lda, n, a_cula, 1.0f, 2.0f);
    memcpy(a_mkl, a_cula, lda*n*sizeof(double));

    // Run CULA's version
    start_time = getHighResolutionTime();
    status = culaDgetrf(m, n, a_cula, lda, ipiv_cula);
    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    // Check CULA errors
    if(status)
    {
        printCulaError(status);
        goto endBenchDgetrf;
    }

    printRuntime(cula_time);

    // Run MKL's version
    start_time = getHighResolutionTime();
    dgetrf(&m, &n, a_mkl, &lda, ipiv_mkl, &info);
    end_time = getHighResolutionTime();
    mkl_time = end_time - start_time;

    if(info != 0)
    {
        printf(" Intel MKL Error: (%d)\n", info);
        goto endBenchDgetrf;
    }

    printRuntime(mkl_time);
    printSpeedup(cula_time, mkl_time);

endBenchDgetrf:
    free(a_cula);
    free(ipiv_cula);
    free(a_mkl);
    free(ipiv_mkl);
    return status;
}
#endif

culaStatus benchSgels(int n)
{
    int m = n + 64;
    int nrhs = 128;

    int lda = m;
    int info = 0;
    int lwork = -1;
    int ldb = imax(m,n);

    float* a_cula = NULL;
    float* a_mkl = NULL;
    float* b_cula = NULL;
    float* b_mkl = NULL;
    float* work_mkl = NULL;

    char trans = 'N';

    double start_time, end_time;
    double cula_time, mkl_time;
    culaStatus status = culaNoError;

    printProblemSize(n);

    a_cula = (float*) malloc(lda*n*sizeof(float));
    a_mkl = (float*) malloc(lda*n*sizeof(float));
    b_cula = (float*) malloc(ldb*nrhs*sizeof(float));
    b_mkl = (float*) malloc(ldb*nrhs*sizeof(float));
    work_mkl = (float*) malloc(1*sizeof(float));

    if(!a_cula || !a_mkl || !b_cula || !b_mkl || !work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchSgels;
    }

    // Add some random data
    fillRandomSingle(lda, n, a_cula, -10.0f, 10.0f);
    fillRandomSingle(ldb, nrhs, b_cula, -10.0f, 10.0f);
    memcpy(a_mkl, a_cula, lda*n*sizeof(float));
    memcpy(b_mkl, b_cula, ldb*nrhs*sizeof(float));

    // Run CULA's version
    start_time = getHighResolutionTime();
    status = culaSgels(trans, m, n, nrhs, a_cula, lda, b_cula, ldb);
    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    // Check CULA errors
    if(status)
    {
        printCulaError(status);
        goto endBenchSgels;
    }

    printRuntime(cula_time);

    // Prepare MKL benchmark
    sgels(&trans, &m, &n, &nrhs, a_mkl, &lda, b_mkl, &ldb, work_mkl, &lwork, &info); // Worksize query
    lwork = (int) work_mkl[0];
    free(work_mkl);
    work_mkl = (float*) malloc(lwork*sizeof(float));

    if(!work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchSgels;
    }

    // Run MKL's version
    start_time = getHighResolutionTime();
    sgels(&trans, &m, &n, &nrhs, a_mkl, &lda, b_mkl, &ldb, work_mkl, &lwork, &info);
    end_time = getHighResolutionTime();
    mkl_time = end_time - start_time;

    if(info != 0)
    {
        printf(" Intel MKL Error: (%d)\n", info);
        goto endBenchSgels;
    }

    printRuntime(mkl_time);
    printSpeedup(cula_time, mkl_time);

endBenchSgels:
    free(a_cula);
    free(b_cula);
    free(a_mkl);
    free(b_mkl);
    free(work_mkl);
    return status;
}

#ifdef CULA_PREMIUM
culaStatus benchDgels(int n)
{
    int m = n + 64;
    int nrhs = 128;

    int lda = m;
    int info = 0;
    int lwork = -1;
    int ldb = imax(m,n);

    double* a_cula = NULL;
    double* a_mkl = NULL;
    double* b_cula = NULL;
    double* b_mkl = NULL;
    double* work_mkl = NULL;

    char trans = 'N';

    double start_time, end_time;
    double cula_time, mkl_time;
    culaStatus status = culaNoError;

    printProblemSize(n);

    a_cula = (double*) malloc(lda*n*sizeof(double));
    a_mkl = (double*) malloc(lda*n*sizeof(double));
    b_cula = (double*) malloc(ldb*nrhs*sizeof(double));
    b_mkl = (double*) malloc(ldb*nrhs*sizeof(double));
    work_mkl = (double*) malloc(1*sizeof(double));

    if(!a_cula || !a_mkl || !b_cula || !b_mkl || !work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchDgels;
    }

    // Add some random data
    fillRandomDouble(lda, n, a_cula, -10.0f, 10.0f);
    fillRandomDouble(ldb, nrhs, b_cula, -10.0f, 10.0f);
    memcpy(a_mkl, a_cula, lda*n*sizeof(double));
    memcpy(b_mkl, b_cula, ldb*nrhs*sizeof(double));

    // Run CULA's version
    start_time = getHighResolutionTime();
    status = culaDgels(trans, m, n, nrhs, a_cula, lda, b_cula, ldb);
    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    // Check CULA errors
    if(status)
    {
        printCulaError(status);
        goto endBenchDgels;
    }

    printRuntime(cula_time);

    // Prepare MKL benchmark
    dgels(&trans, &m, &n, &nrhs, a_mkl, &lda, b_mkl, &ldb, work_mkl, &lwork, &info); // Worksize query
    lwork = (int) work_mkl[0];
    free(work_mkl);
    work_mkl = (double*) malloc(lwork*sizeof(double));

    if(!work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchDgels;
    }

    // Run MKL's version
    start_time = getHighResolutionTime();
    dgels(&trans, &m, &n, &nrhs, a_mkl, &lda, b_mkl, &ldb, work_mkl, &lwork, &info);
    end_time = getHighResolutionTime();
    mkl_time = end_time - start_time;

    if(info != 0)
    {
        printf(" Intel MKL Error: (%d)\n", info);
        goto endBenchDgels;
    }

    printRuntime(mkl_time);
    printSpeedup(cula_time, mkl_time);

endBenchDgels:
    free(a_cula);
    free(b_cula);
    free(a_mkl);
    free(b_mkl);
    free(work_mkl);
    return status;
}
#endif

culaStatus benchSgglse(int n)
{
    int m = n;
    int p = 100;

    int lda = m;
    int info = 0;
    int lwork = -1;
    int ldb = p;

    float* a_cula = NULL;
    float* a_mkl = NULL;
    float* b_cula = NULL;
    float* b_mkl = NULL;
    float* c_cula = NULL;
    float* c_mkl = NULL;
    float* d_cula = NULL;
    float* d_mkl = NULL;
    float* x_cula = NULL;
    float* x_mkl = NULL;
    float* work_mkl = NULL;

    double start_time, end_time;
    double cula_time, mkl_time;
    culaStatus status = culaNoError;

    printProblemSize(n);

    a_cula = (float*) malloc(lda*n*sizeof(float));
    a_mkl = (float*) malloc(lda*n*sizeof(float));
    b_cula = (float*) malloc(ldb*n*sizeof(float));
    b_mkl = (float*) malloc(ldb*n*sizeof(float));
    c_cula = (float*) malloc(m*sizeof(float));
    c_mkl = (float*) malloc(m*sizeof(float));
    d_cula = (float*) malloc(p*sizeof(float));
    d_mkl = (float*) malloc(p*sizeof(float));
    x_cula = (float*) malloc(n*sizeof(float));
    x_mkl = (float*) malloc(n*sizeof(float));
    work_mkl = (float*) malloc(1*sizeof(float));

    if(!a_cula || !a_mkl || !b_cula || !b_mkl || !c_cula || !c_mkl || !d_cula || !d_mkl || !x_cula || !x_mkl || !work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchSgglse;
    }

    // Add some random data
    fillRandomSingle(lda, n, a_cula, -10.0f, 10.0f);
    fillRandomSingle(ldb, n, b_cula, -10.0f, 10.0f);
    fillRandomSingle(1, m, c_cula, -10.0f, 10.0f);
    fillRandomSingle(1, p, d_cula, -10.f, 10.0f);

    memcpy(a_mkl, a_cula, lda*n*sizeof(float));
    memcpy(b_mkl, b_cula, ldb*n*sizeof(float));
    memcpy(c_mkl, c_cula, m*sizeof(float));
    memcpy(d_mkl, d_cula, p*sizeof(float));

    // Run CULA version
    start_time = getHighResolutionTime();
    status = culaSgglse(m, n, p, a_cula, lda, b_cula, ldb, c_cula, d_cula, x_cula);
    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    // Check CULA errors
    if(status)
    {
        printCulaError(status);
        goto endBenchSgglse;
    }

    printRuntime(cula_time);

    // Prepare MKL benchmark
    sgglse(&m, &n, &p, a_mkl, &lda, b_mkl, &ldb, c_mkl, d_mkl, x_mkl, work_mkl, &lwork, &info); // Worksize query
    lwork = (int) work_mkl[0];
    free(work_mkl);
    work_mkl = (float*) malloc(lwork*sizeof(float));

    if(!work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchSgglse;
    }

    // Run MKL's verion
    start_time = getHighResolutionTime();
    sgglse(&m, &n, &p, a_mkl, &lda, b_mkl, &ldb, c_mkl, d_mkl, x_mkl, work_mkl, &lwork, &info);
    end_time = getHighResolutionTime();
    mkl_time = end_time - start_time;

    if(info != 0)
    {
        printf(" Intel MKL Error: (%d)\n", info);
        goto endBenchSgglse;
    }

    printRuntime(mkl_time);
    printSpeedup(cula_time, mkl_time);

endBenchSgglse:
    free(a_cula);
    free(b_cula);
    free(c_cula);
    free(d_cula);
    free(x_cula);
    free(a_mkl);
    free(b_mkl);
    free(c_mkl);
    free(d_mkl);
    free(x_mkl);
    free(work_mkl);
    return status;
}

#ifdef CULA_PREMIUM
culaStatus benchDgglse(int n)
{
    int m = n;
    int p = 100;

    int lda = m;
    int info = 0;
    int lwork = -1;
    int ldb = p;

    double* a_cula = NULL;
    double* a_mkl = NULL;
    double* b_cula = NULL;
    double* b_mkl = NULL;
    double* c_cula = NULL;
    double* c_mkl = NULL;
    double* d_cula = NULL;
    double* d_mkl = NULL;
    double* x_cula = NULL;
    double* x_mkl = NULL;
    double* work_mkl = NULL;

    double start_time, end_time;
    double cula_time, mkl_time;
    culaStatus status = culaNoError;

    printProblemSize(n);

    a_cula = (double*) malloc(lda*n*sizeof(double));
    a_mkl = (double*) malloc(lda*n*sizeof(double));
    b_cula = (double*) malloc(ldb*n*sizeof(double));
    b_mkl = (double*) malloc(ldb*n*sizeof(double));
    c_cula = (double*) malloc(m*sizeof(double));
    c_mkl = (double*) malloc(m*sizeof(double));
    d_cula = (double*) malloc(p*sizeof(double));
    d_mkl = (double*) malloc(p*sizeof(double));
    x_cula = (double*) malloc(n*sizeof(double));
    x_mkl = (double*) malloc(n*sizeof(double));
    work_mkl = (double*) malloc(1*sizeof(double));

    if(!a_cula || !a_mkl || !b_cula || !b_mkl || !c_cula || !c_mkl || !d_cula || !d_mkl || !x_cula || !x_mkl || !work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchDgglse;
    }

    // Add some random data
    fillRandomDouble(lda, n, a_cula, -10.0f, 10.0f);
    fillRandomDouble(ldb, n, b_cula, -10.0f, 10.0f);
    fillRandomDouble(1, m, c_cula, -10.0f, 10.0f);
    fillRandomDouble(1, p, d_cula, -10.f, 10.0f);

    memcpy(a_mkl, a_cula, lda*n*sizeof(double));
    memcpy(b_mkl, b_cula, ldb*n*sizeof(double));
    memcpy(c_mkl, c_cula, m*sizeof(double));
    memcpy(d_mkl, d_cula, p*sizeof(double));

    // Run CULA version
    start_time = getHighResolutionTime();
    status = culaDgglse(m, n, p, a_cula, lda, b_cula, ldb, c_cula, d_cula, x_cula);
    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    // Check CULA errors
    if(status)
    {
        printCulaError(status);
        goto endBenchDgglse;
    }

    printRuntime(cula_time);

    // Prepare MKL benchmark
    dgglse(&m, &n, &p, a_mkl, &lda, b_mkl, &ldb, c_mkl, d_mkl, x_mkl, work_mkl, &lwork, &info); // Worksize query
    lwork = (int) work_mkl[0];
    free(work_mkl);
    work_mkl = (double*) malloc(lwork*sizeof(double));

    if(!work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchDgglse;
    }

    // Run MKL's verion
    start_time = getHighResolutionTime();
    dgglse(&m, &n, &p, a_mkl, &lda, b_mkl, &ldb, c_mkl, d_mkl, x_mkl, work_mkl, &lwork, &info);
    end_time = getHighResolutionTime();
    mkl_time = end_time - start_time;

    if(info != 0)
    {
        printf(" Intel MKL Error: (%d)\n", info);
        goto endBenchDgglse;
    }

    printRuntime(mkl_time);
    printSpeedup(cula_time, mkl_time);

endBenchDgglse:
    free(a_cula);
    free(b_cula);
    free(c_cula);
    free(d_cula);
    free(x_cula);
    free(a_mkl);
    free(b_mkl);
    free(c_mkl);
    free(d_mkl);
    free(x_mkl);
    free(work_mkl);
    return status;
}
#endif

culaStatus benchSgesv(int n)
{
    int nrhs = 1;

    int lda = n;
    int ldb = n;
    int info = 0;

    float* a_cula = NULL;
    float* a_mkl = NULL;
    int* ipiv_cula = NULL;
    int* ipiv_mkl = NULL;
    float* b_cula = NULL;
    float* b_mkl = NULL;

    double start_time, end_time;
    double cula_time, mkl_time;
    culaStatus status = culaNoError;

    printProblemSize(n);

    a_cula = (float*) malloc(lda*n*sizeof(float));
    a_mkl = (float*) malloc(lda*n*sizeof(float));
    ipiv_cula = (int*) malloc(n*sizeof(int));
    ipiv_mkl = (int*) malloc(n*sizeof(int));
    b_cula = (float*) malloc(ldb*nrhs*sizeof(float));
    b_mkl = (float*) malloc(ldb*nrhs*sizeof(float));

    if(!a_cula || !a_mkl || !ipiv_cula || !ipiv_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchSgesv;
    }

    // Add some random data
    fillRandomSingle(lda, n, a_cula, -10.0f, 10.0f);
    fillRandomSingle(ldb, nrhs, b_cula, -10.0f, 10.0f);
    memcpy(a_mkl, a_cula, lda*n*sizeof(float));
    memcpy(b_mkl, b_cula, ldb*nrhs*sizeof(float));

    // Run CULA's version
    start_time = getHighResolutionTime();
    status = culaSgesv(n, nrhs, a_cula, lda, ipiv_cula, b_cula, ldb);
    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    // Check CULA errors
    if(status)
    {
        printCulaError(status);
        goto endBenchSgesv;
    }

    printRuntime(cula_time);

    // Run MKL's version
    start_time = getHighResolutionTime();
    sgesv(&n, &nrhs, a_mkl, &lda, ipiv_mkl, b_mkl, &ldb, &info);
    end_time = getHighResolutionTime();
    mkl_time = end_time - start_time;

    if(info != 0)
    {
        printf(" Intel MKL Error: (%d)\n", info);
        goto endBenchSgesv;
    }

    printRuntime(mkl_time);
    printSpeedup(cula_time, mkl_time);

endBenchSgesv:
    free(a_cula);
    free(ipiv_cula);
    free(a_mkl);
    free(ipiv_mkl);
    free(b_cula);
    free(b_mkl);
    return status;
}

#ifdef CULA_PREMIUM
culaStatus benchDgesv(int n)
{
    int nrhs = 1;

    int lda = n;
    int ldb = n;
    int info = 0;

    double* a_cula = NULL;
    double* a_mkl = NULL;
    int* ipiv_cula = NULL;
    int* ipiv_mkl = NULL;
    double* b_cula = NULL;
    double* b_mkl = NULL;

    double start_time, end_time;
    double cula_time, mkl_time;
    culaStatus status = culaNoError;

    printProblemSize(n);

    a_cula = (double*) malloc(lda*n*sizeof(double));
    a_mkl = (double*) malloc(lda*n*sizeof(double));
    ipiv_cula = (int*) malloc(n*sizeof(int));
    ipiv_mkl = (int*) malloc(n*sizeof(int));
    b_cula = (double*) malloc(ldb*nrhs*sizeof(double));
    b_mkl = (double*) malloc(ldb*nrhs*sizeof(double));

    if(!a_cula || !a_mkl || !ipiv_cula || !ipiv_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchDgesv;
    }

    // Add some random data
    fillRandomDouble(lda, n, a_cula, -10.0f, 10.0f);
    fillRandomDouble(ldb, nrhs, b_cula, -10.0f, 10.0f);
    memcpy(a_mkl, a_cula, lda*n*sizeof(double));
    memcpy(b_mkl, b_cula, ldb*nrhs*sizeof(double));

    // Run CULA's version
    start_time = getHighResolutionTime();
    status = culaDgesv(n, nrhs, a_cula, lda, ipiv_cula, b_cula, ldb);
    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    // Check CULA errors
    if(status)
    {
        printCulaError(status);
        goto endBenchDgesv;
    }

    printRuntime(cula_time);

    // Run MKL's version
    start_time = getHighResolutionTime();
    dgesv(&n, &nrhs, a_mkl, &lda, ipiv_mkl, b_mkl, &ldb, &info);
    end_time = getHighResolutionTime();
    mkl_time = end_time - start_time;

    if(info != 0)
    {
        printf(" Intel MKL Error: (%d)\n", info);
        goto endBenchDgesv;
    }

    printRuntime(mkl_time);
    printSpeedup(cula_time, mkl_time);

endBenchDgesv:
    free(a_cula);
    free(ipiv_cula);
    free(a_mkl);
    free(ipiv_mkl);
    free(b_cula);
    free(b_mkl);
    return status;
}
#endif

culaStatus benchSgesvd(int n)
{
    int m = n;

    char jobu = 'A';
    char jobvt = 'A';

    int lda = m;
    int ldu = m;
    int ldvt = n;
    int ucol = imin(m,n);

    int info = 0;
    int lwork = -1;

    float *a_cula = NULL;
    float *a_mkl = NULL;
    float *s_cula = NULL;
    float *s_mkl = NULL;
    float *u_cula = NULL;
    float *u_mkl = NULL;
    float *vt_cula = NULL;
    float *vt_mkl = NULL;
    float *work_mkl = NULL;

    double start_time, end_time;
    double cula_time, mkl_time;
    culaStatus status = culaNoError;

    printProblemSize(n);

    a_cula = (float*) malloc(lda*n*sizeof(float));
    a_mkl = (float*) malloc(lda*n*sizeof(float));
    s_cula = (float*) malloc(imin(m,n)*sizeof(float));
    s_mkl = (float*) malloc(imin(m,n)*sizeof(float));
    u_cula = (float*) malloc(ldu*ucol*sizeof(float));
    u_mkl = (float*) malloc(ldu*ucol*sizeof(float));
    vt_cula = (float*) malloc(ldvt*n*sizeof(float));
    vt_mkl = (float*) malloc(ldvt*n*sizeof(float));
    work_mkl = (float*) malloc(1*sizeof(float));

    if(!a_cula || !a_mkl || !s_cula || !s_mkl || !u_cula || !u_mkl || !vt_cula || !vt_mkl || !work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchSgesvd;
    }

    fillRandomSingle(lda, n, a_cula, 1.0f, 256.0f);
    memcpy(a_mkl, a_cula, lda*n*sizeof(float));

    // Run CULA version
    start_time = getHighResolutionTime();
    status = culaSgesvd(jobu, jobvt, m, n, a_cula, lda, s_cula, u_cula, ldu, vt_cula, ldvt);
    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    // Check CULA errors
    if(status)
    {
        printCulaError(status);
        goto endBenchSgesvd;
    }

    printRuntime(cula_time);

    // Prepare MKL benchmark
    sgesvd(&jobu, &jobvt, &m, &n, a_mkl, &lda, s_mkl, u_mkl, &ldu, vt_mkl, &ldvt, work_mkl, &lwork, &info); // Workspace query
    lwork = (int) work_mkl[0];
    free(work_mkl);
    work_mkl = (float*) malloc(lwork*sizeof(float));

    if(!work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchSgesvd;
    }

    // Run MKL's verion
    start_time = getHighResolutionTime();
    sgesvd(&jobu, &jobvt, &m, &n, a_mkl, &lda, s_mkl, u_mkl, &ldu, vt_mkl, &ldvt, work_mkl, &lwork, &info);
    end_time = getHighResolutionTime();
    mkl_time = end_time - start_time;

    if(info != 0)
    {
        printf(" Intel MKL Error: (%d)\n", info);
        goto endBenchSgesvd;
    }

    printRuntime(mkl_time);
    printSpeedup(cula_time, mkl_time);

endBenchSgesvd:
    free(a_cula);
    free(a_mkl);
    free(s_cula);
    free(s_mkl);
    free(u_cula);
    free(u_mkl);
    free(vt_cula);
    free(vt_mkl);
    free(work_mkl);
    return status;
}

#ifdef CULA_PREMIUM
culaStatus benchDgesvd(int n)
{
    int m = n;

    char jobu = 'A';
    char jobvt = 'A';

    int lda = m;
    int ldu = m;
    int ldvt = n;
    int ucol = imin(m,n);

    int info = 0;
    int lwork = -1;

    double *a_cula = NULL;
    double *a_mkl = NULL;
    double *s_cula = NULL;
    double *s_mkl = NULL;
    double *u_cula = NULL;
    double *u_mkl = NULL;
    double *vt_cula = NULL;
    double *vt_mkl = NULL;
    double *work_mkl = NULL;

    double start_time, end_time;
    double cula_time, mkl_time;
    culaStatus status = culaNoError;

    printProblemSize(n);

    a_cula = (double*) malloc(lda*n*sizeof(double));
    a_mkl = (double*) malloc(lda*n*sizeof(double));
    s_cula = (double*) malloc(imin(m,n)*sizeof(double));
    s_mkl = (double*) malloc(imin(m,n)*sizeof(double));
    u_cula = (double*) malloc(ldu*ucol*sizeof(double));
    u_mkl = (double*) malloc(ldu*ucol*sizeof(double));
    vt_cula = (double*) malloc(ldvt*n*sizeof(double));
    vt_mkl = (double*) malloc(ldvt*n*sizeof(double));
    work_mkl = (double*) malloc(1*sizeof(double));

    if(!a_cula || !a_mkl || !s_cula || !s_mkl || !u_cula || !u_mkl || !vt_cula || !vt_mkl || !work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchDgesvd;
    }

    fillRandomDouble(lda, n, a_cula, 1.0f, 256.0f);
    memcpy(a_mkl, a_cula, lda*n*sizeof(double));

    // Run CULA version
    start_time = getHighResolutionTime();
    status = culaDgesvd(jobu, jobvt, m, n, a_cula, lda, s_cula, u_cula, ldu, vt_cula, ldvt);
    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    // Check CULA errors
    if(status)
    {
        printCulaError(status);
        goto endBenchDgesvd;
    }

    printRuntime(cula_time);

    // Prepare MKL benchmark
    dgesvd(&jobu, &jobvt, &m, &n, a_mkl, &lda, s_mkl, u_mkl, &ldu, vt_mkl, &ldvt, work_mkl, &lwork, &info); // Workspace query
    lwork = (int) work_mkl[0];
    free(work_mkl);
    work_mkl = (double*) malloc(lwork*sizeof(double));

    if(!work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchDgesvd;
    }

    // Run MKL's verion
    start_time = getHighResolutionTime();
    dgesvd(&jobu, &jobvt, &m, &n, a_mkl, &lda, s_mkl, u_mkl, &ldu, vt_mkl, &ldvt, work_mkl, &lwork, &info);
    end_time = getHighResolutionTime();
    mkl_time = end_time - start_time;

    if(info != 0)
    {
        printf(" Intel MKL Error: (%d)\n", info);
        goto endBenchDgesvd;
    }

    printRuntime(mkl_time);
    printSpeedup(cula_time, mkl_time);

endBenchDgesvd:
    free(a_cula);
    free(a_mkl);
    free(s_cula);
    free(s_mkl);
    free(u_cula);
    free(u_mkl);
    free(vt_cula);
    free(vt_mkl);
    free(work_mkl);
    return status;
}
#endif

#ifdef CULA_PREMIUM
culaStatus benchSsyev(int n)
{
    char jobz = 'N';
    char uplo = 'U';

    int lda = n;
   
    int info = 0;
    int lwork = -1;

    float *a_cula = NULL;
    float *a_mkl = NULL;
    float *w_cula = NULL;
    float *w_mkl = NULL;
    float *work_mkl = NULL;

    double start_time, end_time;
    double cula_time, mkl_time;
    culaStatus status = culaNoError;

    printProblemSize(n);

    a_cula = (float*) malloc(lda*n*sizeof(float));
    a_mkl = (float*) malloc(lda*n*sizeof(float));
    w_cula = (float*) malloc(n*sizeof(float));
    w_mkl = (float*) malloc(n*sizeof(float));
    work_mkl = (float*) malloc(1*sizeof(float));

    if(!a_cula || !a_mkl || !w_cula || !w_mkl || !work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchSsyev;
    }

    fillRandomSingle(lda, n, a_cula, 1.0f, 256.0f);
    memcpy(a_mkl, a_cula, lda*n*sizeof(float));

    // Run CULA version
    start_time = getHighResolutionTime();
    status = culaSsyev(jobz, uplo, n, a_cula, lda, w_cula );
    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    // Check CULA errors
    if(status)
    {
        printCulaError(status);
        goto endBenchSsyev;
    }

    printRuntime(cula_time);

    // Prepare MKL benchmark
    ssyev(&jobz, &uplo, &n, a_mkl, &lda, w_mkl, work_mkl, &lwork, &info); // Workspace query
    lwork = (int) work_mkl[0];
    free(work_mkl);
    work_mkl = (float*) malloc(lwork*sizeof(float));

    if(!work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchSsyev;
    }

    // Run MKL's verion
    start_time = getHighResolutionTime();
    ssyev(&jobz, &uplo, &n, a_mkl, &lda, w_mkl, work_mkl, &lwork, &info);
    end_time = getHighResolutionTime();
    mkl_time = end_time - start_time;

    if(info != 0)
    {
        printf(" Intel MKL Error: (%d)\n", info);
        goto endBenchSsyev;
    }

    printRuntime(mkl_time);
    printSpeedup(cula_time, mkl_time);

endBenchSsyev:
    free(a_cula);
    free(a_mkl);
    free(w_cula);
    free(w_mkl);
    free(work_mkl);
    return status;
}
#endif

#ifdef CULA_PREMIUM
culaStatus benchDsyev(int n)
{
    char jobz = 'N';
    char uplo = 'U';

    int lda = n;
   
    int info = 0;
    int lwork = -1;

    double *a_cula = NULL;
    double *a_mkl = NULL;
    double *w_cula = NULL;
    double *w_mkl = NULL;
    double *work_mkl = NULL;

    double start_time, end_time;
    double cula_time, mkl_time;
    culaStatus status = culaNoError;

    printProblemSize(n);

    a_cula = (double*) malloc(lda*n*sizeof(double));
    a_mkl = (double*) malloc(lda*n*sizeof(double));
    w_cula = (double*) malloc(n*sizeof(double));
    w_mkl = (double*) malloc(n*sizeof(double));
    work_mkl = (double*) malloc(1*sizeof(double));

    if(!a_cula || !a_mkl || !w_cula || !w_mkl || !work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchDsyev;
    }

    fillRandomDouble(lda, n, a_cula, 1.0, 256.0);
    memcpy(a_mkl, a_cula, lda*n*sizeof(double));

    // Run CULA version
    start_time = getHighResolutionTime();
    status = culaDsyev(jobz, uplo, n, a_cula, lda, w_cula );
    end_time = getHighResolutionTime();
    cula_time = end_time - start_time;

    // Check CULA errors
    if(status)
    {
        printCulaError(status);
        goto endBenchDsyev;
    }

    printRuntime(cula_time);

    // Prepare MKL benchmark
    dsyev(&jobz, &uplo, &n, a_mkl, &lda, w_mkl, work_mkl, &lwork, &info); // Workspace query
    lwork = (int) work_mkl[0];
    free(work_mkl);
    work_mkl = (double*) malloc(lwork*sizeof(double));

    if(!work_mkl)
    {
        printf(" Host side allocation error.\n");
        status = culaInsufficientMemory;
        goto endBenchDsyev;
    }

    // Run MKL's verion
    start_time = getHighResolutionTime();
    dsyev(&jobz, &uplo, &n, a_mkl, &lda, w_mkl, work_mkl, &lwork, &info);
    end_time = getHighResolutionTime();
    mkl_time = end_time - start_time;

    if(info != 0)
    {
        printf(" Intel MKL Error: (%d)\n", info);
        goto endBenchDsyev;
    }

    printRuntime(mkl_time);
    printSpeedup(cula_time, mkl_time);

endBenchDsyev:
    free(a_cula);
    free(a_mkl);
    free(w_cula);
    free(w_mkl);
    free(work_mkl);
    return status;
}
#endif

void initializeMkl()
{}

void printCulaError(culaStatus status)
{
    char buf[256];

    if(!status)
        return;

    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    printf("\n%s\n", buf);
}

