#
# Feel free to modify and polish the Makefile if you wish.
#
# Also, you may (and should) add more compiler flags when introducing some
#   optimization techniques. BUT, it is not allowed to turn off `-W*` error
#   flags. Be strict on warnings is a good habit.
#
# Jose @ ShanghaiTech University
#

# Adapt this makefile to macos
SYSNAME := $(shell uname -s)
ifeq ($(SYSNAME), Linux)
	CC=g++
endif
ifeq ($(SYSNAME), Darwin)
	CC=g++-9
endif

CFLAGS=-Wpedantic -Wall -Wextra -Werror -O3 -fopenmp -std=c++11

all: kmeans

kmeans: kmeans.cpp kmeans.h
	${CC} ${CFLAGS} kmeans.cpp -o kmeans

benchmark: benchmark.cpp kmeans.h
	${CC} ${CFLAGS} benchmark.cpp -o benchmark

.PHONY: clean gen plot

clean:
	rm -f kmeans

gen: generate.py
	python3 generate.py ${FILE}

plot: plot.py
	python3 plot.py ${FILE}

submission:
	tar czvf project3.tar .git
