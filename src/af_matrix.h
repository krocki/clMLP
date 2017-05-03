/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-05-01 22:22:38
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-05-02 20:46:27
*/

#include <containers/memarray.h>
#include <arrayfire.h>

#ifndef __ARRAYFIRE_ARRAY__
#define __ARRAYFIRE_ARRAY__

// !TODO!

template <typename T = float>
class afire_array : public memarray<T> {

};


// zeros: array zeros = constant (0,5) 1 x 5 zeros
// zeros2d: array zeros2d = constant (0,2,3) 2 x 3 zeros

// reshape
// array train_data = moddims(train_images, 784, 60000)
// array train_images = moddims(train_images, 28, 28, 60000)

// y = matmul(w, x)
// array sigmoid(const array& val) { return 1/(1+exp(-val);)}
// err = t - p
// w += learning_rate * (matmulNT(err, data_train))


#endif