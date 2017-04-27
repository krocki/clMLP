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
cmake .. -DCMAKE_INSTALL_PATH=/opt/clBlas -DCMAKE_BUILD_TYPE=Release -DPRECOMPILE_GEMM_PRECISION_SGEMM:BOOL=ON -DPRECOMPILE_GEMM_TRANS_NN:BOOL=ON -DPRECOMPILE_GEMM_TRANS_NT:BOOL=ON-DPRECOMPILE_GEMM_TRANS_TN:BOOL=ON -DPRECOMPILE_GEMM_TRANS_TT:BOOL=ON ..
make
sudo make install

```

#clRNG
https://github.com/clMathLibraries/clRNG

# Optional

CLBlast
```
https://github.com/CNugteren/CLBlast
```

CLTune
```
https://github.com/CNugteren/CLTune
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
