/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-03-03 15:06:37
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-25 15:53:24
*/

#ifndef __CL_LAYERS_H__
#define __CL_LAYERS_H__

#include <nn/clnn_utils.h>
#include <opencl/cl_matrix.h>
#include <opencl/cl_functions.h>

//abstract
class CLLayer {

  public:

	//used in forward pass
	cl_matrix x, y, dx, dy;

	CLLayer(cl_ctx* cl_env, size_t inputs, size_t outputs, size_t batch_size) {

		x = cl_matrix(cl_env, {inputs, batch_size});
		y = cl_matrix(cl_env, {outputs, batch_size});
		dx = cl_matrix(cl_env, {inputs, batch_size});
		dy = cl_matrix(cl_env, {outputs, batch_size});

	};

	//need to override these
	virtual void forward() = 0;
	virtual void backward() = 0;
	virtual void resetGrads() {};
	virtual void applyGrads(float alpha) { UNUSED(alpha); };

	virtual void sync_device() {}; // sync params h -> d
	virtual void sync_host() {}; // sync params d -> h

	virtual ~CLLayer() {};

};

class Linear : public CLLayer {

  public:

	cl_matrix W, dW;

	void forward() {

		cl_matrix_mult(y, W, x, false, false, 1.0f, 0.0f);

	}

	void backward() {

		cl_matrix_mult(dW, dy, x, false, true, 1.0f, 0.0f);
		cl_matrix_mult(dx, W, dy, true, false, 1.0f, 0.0f);

	}

	Linear(cl_ctx* cl_env, size_t inputs, size_t outputs, size_t batch_size) : CLLayer(cl_env, inputs, outputs, batch_size) {

		W = cl_matrix(cl_env, {outputs, inputs});
		dW = cl_matrix(W);

		matrix_randn ( W.ref_host_data, 0, ( 1.0f ) / sqrtf ( W.ref_host_data.rows() + W.ref_host_data.cols() ) );
		// cl_elementwise(W, "randn", true);

		W.sync_device();

	};

	void resetGrads() {

		dW.setZero();

	}

	virtual void sync_device() {

		W.sync_device();

	};

	virtual void sync_host() {

		W.sync_host();

	};

	void applyGrads(float alpha) {

		cl_elementwise(W, dW, alpha, "fmad", true);

	}

	~Linear() {};

};

class Sigmoid : public CLLayer {

  public:

	void forward() {

		cl_elementwise(y, x, "logistic", true);

	}

	void backward() {

		cl_elementwise(dx, dy, y, "dlogistic", true);

	}

	Sigmoid(cl_ctx* cl_env, size_t inputs, size_t outputs, size_t batch_size) : CLLayer(cl_env, inputs, outputs, batch_size) {};
	~Sigmoid() {};

};

class ReLU : public CLLayer {

  public:

	void forward() {

		cl_elementwise(y, x, "relu", true);

	}

	void backward() {

		cl_elementwise(dx, dy, y, "drelu", true);

	}

	ReLU(cl_ctx* cl_env, size_t inputs, size_t outputs, size_t batch_size) : CLLayer(cl_env, inputs, outputs, batch_size) {};
	~ReLU() {};

};

class Softmax : public CLLayer {

  public:

	void forward() {

		cl_sub_max_coeff(x, true);
		cl_softmax(y, x, true);


	}

	void backward() {

		cl_elementwise(dx, dy, y, "dsoftmax", true);

	}


	Softmax(cl_ctx* cl_env, size_t inputs, size_t outputs, size_t batch_size) : CLLayer(cl_env, inputs, outputs, batch_size) {};
	~Softmax() {};

};

#endif