# Makefile for benchmark example
#
# To compile this benchmark, please ensure that some extra environment
# variables are defined. An example configuration may look like the following:
#
#     export INTEL_ROOT=/opt/intel/composerxe-2011.3.174
#     export INTEL_LIB_PATH_32=$INTEL_ROOT/compiler/lib/ia32
#     export INTEL_LIB_PATH_64=$INTEL_ROOT/compiler/lib/intel64
#
#     export MKL_ROOT=$INTEL_ROOT/mkl
#     export MKL_INC_PATH=$MKL_ROOT/include
#     export MKL_LIB_PATH_32=$MKL_ROOT/lib/ia32
#     export MKL_LIB_PATH_64=$MKL_ROOT/lib/intel64
#
# Due to recent changes in Intel's MKL distribution, this example only supports
# MKL 10.3 or greater. 


CC=gcc
FC=ifort
ifdef debug
CFLAGS=-g
else
CFLAGS=-DNDEBUG -O3
endif

INCLUDES=-I$(MKLROOT)/include
LDFLAGS=-L$(MKLROOT)/lib/intel64

INCLUDES=-I${CULA_INC_PATH} -I${MKLROOT}/include
LIBPATH64=-L${CULA_LIB_PATH_64} -L${MKLROOT}/../compiler/lib/intel64 -L${MKLROOT}/lib/intel64

LIBS=-lcula_lapack_basic -lcublas -lpthread
MKL_LIBS_64=-lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5

usage:
	@echo "To build this example, type one of:"
	@echo ""
	@echo "    make build32"
	@echo "    make build64"
	@echo ""
	@echo "where '32' and '64' represent the platform you wish to build for"
	@echo ""
	@echo "Note that this example requires Intel MKL 10.3 or greater; if you do not have"
	@echo "MKL, or your version of MKL is insufficient, you may use the precompiled"
	@echo "binary, 'benchmark_', found in this directory."

build32:
#	sh ../checkenvironment.sh
	${CC} -m32 -o benchmark benchmark.c $(CFLAGS) $(INCLUDES) $(LIBPATH32) $(MKL_LIBS_32) $(LIBS)

build64:
#	sh ../checkenvironment.sh
	${CC} -m64 -o benchmark benchmark.c $(CFLAGS) $(INCLUDES) $(LIBPATH64) $(MKL_LIBS_64) $(LIBS)

clean:
	rm -f benchmark

