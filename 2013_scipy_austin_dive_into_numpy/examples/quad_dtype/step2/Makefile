CC	= gcc
LD	= gcc

CFLAGS	= -DDEBUG -g -O0 -Wall -W -I/home/vagrant/src/numpy/numpy-git/numpy/core/include -I/usr/include/python2.7
LDFLAGS	= -g -shared -lquadmath

OBJECTS = _quad.o number_protocol.o python_wrapper.o

_quad.so: $(OBJECTS)
	$(LD) -o $@ $(OBJECTS) $(LDFLAGS)

.c.o:
	$(CC) -c $< -o $@ $(CFLAGS)

clean:
	rm -f *.o
	rm -f _quad.so
