CC=gcc

INCLUDES=-I$(MKLROOT)/include
LDFLAGS=-L$(MKLROOT)/lib/intel64

LIBS=-lcuda -lmagma  -lm
OBJS=dgesv-gpu.o
EXEC=dgesv-gpu

all: $(EXEC) 

$(EXEC): $(OBJS)
	$(CC) $(LDFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

%.o: %.c
	$(CC) -O3 -DADD_ -DCUBLAS_GFORTRAN -DHAVE_CUBLAS $(INCLUDES) -c $< 

clean:
	$(RM) *.o *.so *.a $(EXEC)
