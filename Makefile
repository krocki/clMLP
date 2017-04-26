OS := $(shell uname)

CC=g++

CFLAGS = -Ofast -Wall -Wextra -Wfatal-errors -std=c++14
LFLAGS = -lclBLAS
EIGEN_PATH = /usr/local/Cellar/eigen/3.3.3/include/eigen3/
INCLUDES = -I./src/ -I$(EIGEN_PATH)

#CLBLAS
LFLAGS := $(LFLAGS) -lclBLAS

ifeq ($(OS),Linux)
		CL_LIB_PATH := /usr/local/cuda-8.0/lib
		CL_INCLUDE_PATH := /usr/local/cuda-8.0/include
        LFLAGS := $(LFLAGS) -lOpenCL
        LFLAGS := $(LFLAGS) -L$(CL_LIB_PATH)
        INCLUDES := $(INCLUDES) -I$(CL_INCLUDE_PATH)
else
        #OSX
        LFLAGS := $(LFLAGS) -framework OpenCL
endif

all:
	$(CC) -o clmlp ./src/clmlp.cc $(INCLUDES) $(CFLAGS) $(LFLAGS)
