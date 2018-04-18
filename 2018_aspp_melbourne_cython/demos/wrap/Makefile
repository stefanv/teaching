.PHONY: all

all:
	g++ -c Circle.cpp
	g++ -Wall -o test_circle test_circle.cpp Circle.o

test:
	@echo "Running C++ test:"
	@./test_circle
	@echo "\nRunning Python test:"
	@python ./test_circle.py
