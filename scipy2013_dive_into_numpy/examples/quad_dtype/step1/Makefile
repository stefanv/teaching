CC	= gcc

CFLAGS	= -DDEBUG -g -O0 -Wall -W -I/home/vagrant/src/numpy/numpy-git/numpy/core/include -I/usr/include/python2.7
LDFLAGS	= -g -shared -L/usr/lib/gcc/i686-linux-gnu/4.7/ -lquadmath

_quad.so: _quad.o
	$(LD) $< -o $@ $(LDFLAGS)

_quad.o: _quad.c
	$(CC) -c $< -o $@ $(CFLAGS)

clean:
	rm -f *.o
	rm -f _quad.so
