# clMLP
MultiLayer Perceptron in OpenCL

# basic usage
```
./clmlp <device_number>
```




tested with clblas-2.6 and Eigen 3.3.3
on AMD (D700), NVIDIA (Titan X) and Intel (Iris) GPUs

# installation
Requires CLBlas, clRNG and Eigen

# OS X:
```
brew install clblas
brew install eigen
```
# Ubuntu:
https://github.com/amd/OpenCL-caffe/wiki/How-to-set-up-clBLAS-and-OpenCL
```
sudo apt-get install libeigen3-dev
```
clBLAS
```
git clone git@github.com:clMathLibraries/clBLAS.git
cd clBLAS/src
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DPRECOMPILE_GEMM_PRECISION_SGEMM:BOOL=ON -DPRECOMPILE_GEMM_TRANS_NN:BOOL=ON -DPRECOMPILE_GEMM_TRANS_NT:BOOL=ON -DPRECOMPILE_GEMM_TRANS_TN:BOOL=ON -DPRECOMPILE_GEMM_TRANS_TT:BOOL=ON -G "Unix Makefiles" ../src/
make
sudo make install

```

# clRNG: OpenCL Random Number Generator

```
git clone https://github.com/clMathLibraries/clRNG
```

# clProbDist: probability distributions for OpenCL

```
git clone https://github.com/umontreal-simul/clProbDist
```

# Optional

CLBlast

```
git clone https://github.com/CNugteren/CLBlast
```

CLTune

```
git clone https://github.com/CNugteren/CLTune
```

# default

```
make
```
# clBlas
```
make CL_BLAS_IMPL=CLBLAS
```
# clBLAST
```
make CL_BLAS_IMPL=CLBLAST
```
