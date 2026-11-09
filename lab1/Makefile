CXX = g++
CXXFLAGS = -O1 -std=c++11 -Wall -Wextra -Wno-unknown-pragmas
LDLIBS = -lm

all: serial openmp

serial: src/solver.cpp Makefile
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

openmp: src/solver.cpp Makefile
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@ $(LDLIBS)

test: tests/check
	./tests/check

tests/check: tests/check.cpp src/solver.cpp Makefile
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@ $(LDLIBS)

clean:
	rm -f serial openmp tests/check

.PHONY: all test clean
