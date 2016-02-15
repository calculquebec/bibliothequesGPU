CC=gcc

INCLUDES=-I$(MKLROOT)/include
LDFLAGS=-L$(MKLROOT)/lib/intel64

LIBS=-Wl,--start-group -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -Wl,--end-group -lpthread -lm 
OBJS=dgemm.o
EXEC=dgemm

all: $(EXEC) 

$(EXEC): $(OBJS)
	$(CC) $(LDFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

%.o: %.c
	$(CC) $(INCLUDES) -c $< 

clean:
	$(RM) *.o *.so *.a $(EXEC)
