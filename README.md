# clMLP
MultiLayer Perceptron in OpenCL

# basic usage
```
./clmlp <device_number>
```

# installation
Requires CLBlas and Eigen
# OS X:
```
brew install clblas
brew install eigen
```
# Ubuntu:
```
sudo apt-get install libeigen3-dev
```
#requires CLblas and Eigen
tested with clblas-2.6 and Eigen 3.3.3

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
