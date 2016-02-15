CC=nvcc
#CXXFLAGS=-O2

INCLUDES=
LDFLAGS=

LIBS=-lcufft
OBJS=fft-gpu.o
EXEC=fft-gpu

all: $(EXEC) 

$(EXEC): $(OBJS)
	$(CC) $(LDFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

%.o: %.cu
	$(CC) $(INCLUDES) -c $< 

clean:
	$(RM) *.o *.so *.a $(EXEC)
