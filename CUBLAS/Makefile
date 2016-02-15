CC=nvcc
#CXXFLAGS=-O2

INCLUDES=-I$(CUDA_HOME)/include
LDFLAGS=-L$(CUDA_HOME)/lib64

LIBS=-lcublas
OBJS=dgemm-gpu.o
EXEC=dgemm-gpu

all: $(EXEC) 

$(EXEC): $(OBJS)
	$(CC) $(LDFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

%.o: %.cu
	$(CC) $(INCLUDES) -c $< 

clean:
	$(RM) *.o *.so *.a $(EXEC)
