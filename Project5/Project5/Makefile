CC=gcc
CFLAGS=-Iinclude
DEPS = args.h bitmap.h convolution.h
OBJ = args.c bitmap.c convolution.c main.c
LIBS=-framework OpenCL

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

convolution: $(OBJ)
	gcc -o $@ $^ $(CFLAGS) $(LIBS)
