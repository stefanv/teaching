.PHONY: all clean

all: calcmod/calcmod.so

calcmod/calcmod.so:
	python setup.py build_ext -i

clean:
	rm -f calcmod/*.so

test: calcmod/calcmod.so
	nosetests

