OS := $(shell uname)

CC=g++

CFLAGS = -Ofast -Wall -Wextra -Wfatal-errors -std=c++14
LFLAGS = -lclBLAS
EIGEN_PATH = /usr/local/Cellar/eigen/3.3.3/include/eigen3/
INCLUDES = -I./src/ -I$(EIGEN_PATH)

#CLBLAS
LFLAGS := $(LFLAGS) -lclBLAS

ifeq ($(OS),Linux)
        LFLAGS := $(LFLAGS) -lOpenCL
else
        #OSX
        LFLAGS := $(LFLAGS) -framework OpenCL
endif

all:
	$(CC) -o clmlp ./src/clmlp.cc $(INCLUDES) $(CFLAGS) $(LFLAGS)
