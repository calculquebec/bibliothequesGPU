CC=gcc

INCLUDES=-I$(CULA_INC_PATH)
LDFLAGS=-L$(CULA_LIB_PATH_64)

LIBS=-lcula_lapack_basic -lcublas -liomp5
OBJS=sgesv-gpu.o
EXEC=sgesv-gpu

all: $(EXEC) 

$(EXEC): $(OBJS)
	$(CC) $(LDFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

%.o: %.c
	$(CC) $(INCLUDES) -c $< 

clean:
	$(RM) *.o *.so *.a $(EXEC)
