OS := $(shell uname)

CC=g++

CFLAGS = -Ofast -Wall -Wextra -Wfatal-errors -std=c++14
LFLAGS =
EIGEN_PATH = /usr/local/Cellar/eigen/3.3.3/include/eigen3/
INCLUDES = -I./src/ -I$(EIGEN_PATH)

# CL_BLAS_IMPL=CLBLAST
CL_BLAS_IMPL=CLBLAS
CL_TUNE=0

ifeq ($(CL_BLAS_IMPL),CLBLAS)
#CLBLAS
LFLAGS := $(LFLAGS) -lclBLAS
CFLAGS := $(CFLAGS) -DCLBLAS
endif

ifeq ($(CL_BLAS_IMPL),CLBLAST)
#CLBLAST
LFLAGS := $(LFLAGS) -lclblast
CFLAGS := $(CFLAGS) -DCLBLAST
endif

ifeq ($(CL_TUNE),1)
LFLAGS := $(LFLAGS) -lcltune
CFLAGS := $(CFLAGS) -DCLTUNE
endif

#CLRNG
LFLAGS := $(LFLAGS) -lclrng

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
	$(CC) -o clrand ./src/clrand.cc $(INCLUDES) $(CFLAGS) $(LFLAGS)
