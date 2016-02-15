CC=gcc
#CXXFLAGS=-O2

INCLUDES=
LDFLAGS=

LIBS=-lfftw3
OBJS=fft.o
EXEC=fft

all: $(EXEC) 

$(EXEC): $(OBJS)
	$(CC) $(LDFLAGS) -o $(EXEC) $(OBJS) $(LIBS)

%.o: %.c
	$(CC) $(INCLUDES) -c $< 

clean:
	$(RM) *.o *.so *.a $(EXEC)
