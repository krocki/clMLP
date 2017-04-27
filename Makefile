OS := $(shell uname)

CXX=g++
GCC=gcc

CFLAGS = -Ofast -Wall -Wextra -Wfatal-errors -std=c++14
LFLAGS =
EIGEN_PATH = /usr/local/Cellar/eigen/3.3.3/include/eigen3/
INCLUDES = -I./src/ -I$(EIGEN_PATH)

# CL_BLAS_IMPL=CLBLAST
CL_BLAS_IMPL=CLBLAS
CL_TUNE=0

ifeq ($(CL_BLAS_IMPL),CLBLAS)
#CLBLAS
LFLAGS := -lclBLAS $(LFLAGS)
CFLAGS := -DCLBLAS $(CFLAGS)
endif

ifeq ($(CL_BLAS_IMPL),CLBLAST)
#CLBLAST
LFLAGS := -lclblast $(LFLAGS)
CFLAGS := -DCLBLAST $(CFLAGS)
endif

ifeq ($(CL_TUNE),1)
LFLAGS := -lcltune $(LFLAGS)
CFLAGS := -DCLTUNE $(CFLAGS)
endif

#clProbDist
LFLAGS := -lclProbDist $(LFLAGS)

#clRNG
LFLAGS := -lclRNG $(LFLAGS)

ifeq ($(OS),Linux)
		CL_LIB_PATH := /usr/local/cuda-8.0/lib
		CL_INCLUDE_PATH := /usr/local/cuda-8.0/include
  		LFLAGS := $(LFLAGS) -lOpenCL
  		LFLAGS := $(LFLAGS) -L$(CL_LIB_PATH)
  		INCLUDES := $(INCLUDES) -I$(CL_INCLUDE_PATH)
else
        #OSX
        LFLAGS := -framework OpenCL $(LFLAGS)
endif

all:
	$(CXX) -o clmlp ./src/clmlp.cc $(INCLUDES) $(CFLAGS) $(LFLAGS)
	#$(CXX) -o clrand ./src/clrand.cc $(INCLUDES) $(CFLAGS) $(LFLAGS)
	#$(CXX) -o clprobdist ./src/clprobdist.cc $(INCLUDES) $(CFLAGS) $(LFLAGS)
