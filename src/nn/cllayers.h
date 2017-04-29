/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-03-03 15:06:37
* @Last Modified by:   Kamil Rocki
* @Last Modified time: 2017-04-28 19:51:47
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
		cl_matrix<float> x, y, dx, dy;
		
		CLLayer ( cl_ctx *cl_env, size_t inputs, size_t outputs, size_t batch_size ) {
		
			x = cl_matrix<float> ( cl_env, {inputs, batch_size} );
			y = cl_matrix<float> ( cl_env, {outputs, batch_size} );
			dx = cl_matrix<float> ( cl_env, {inputs, batch_size} );
			dy = cl_matrix<float> ( cl_env, {outputs, batch_size} );
			
		};
		
		//need to override these
		virtual void forward() = 0;
		virtual void backward() = 0;
		virtual void resetGrads() {};
		virtual void applyGrads ( float alpha ) { UNUSED ( alpha ); };
		
		virtual void sync_device() {}; // sync params h -> d
		virtual void sync_host() {}; // sync params d -> h
		
		virtual ~CLLayer() {};
		
};

class Linear : public CLLayer {

	public:
	
		cl_matrix<float> W, dW;
		
		void forward() {
		
			_TIMED_CALL_ ( cl_matrix_mult ( y, W, x, false, false, 1.0f, 0.0f ) );
			
		}
		
		void backward() {
		
			_TIMED_CALL_ ( cl_matrix_mult ( dW, dy, x, false, true, 1.0f, 0.0f ) );
			_TIMED_CALL_ ( cl_matrix_mult ( dx, W, dy, true, false, 1.0f, 0.0f ) );
			
		}
		
		Linear ( cl_ctx *cl_env, size_t inputs, size_t outputs, size_t batch_size ) : CLLayer ( cl_env, inputs, outputs, batch_size ) {
		
			W = cl_matrix<float> ( cl_env, {outputs, inputs} );
			dW = cl_matrix<float> ( W );
			
			matrix_randn_host ( W.ref_host_data, 0, ( 1.0f ) / sqrtf ( W.ref_host_data.rows() + W.ref_host_data.cols() ) );
			// cl_elementwise(W, "randn", true);
			
			_TIMED_CALL_ ( W.sync_device() );
			
		};
		
		virtual void sync_device() {
		
			_TIMED_CALL_ ( W.sync_device() );
			
		};
		
		virtual void sync_host() {
		
			_TIMED_CALL_ ( W.sync_host() );
			
		};
		
		void applyGrads ( float alpha ) {
		
			_TIMED_CALL_ ( cl_elementwise ( W, dW, alpha, "fmad_lmem" ) );
			
		}
		
		~Linear() {};
		
};

class Sigmoid : public CLLayer {

	public:
	
		void forward() {
		
			_TIMED_CALL_ ( cl_elementwise ( y, x, "logistic" ) );
			
		}
		
		void backward() {
		
			_TIMED_CALL_ ( cl_elementwise ( dx, dy, y, "dlogistic" ) );
			
		}
		
		Sigmoid ( cl_ctx *cl_env, size_t inputs, size_t outputs, size_t batch_size ) : CLLayer ( cl_env, inputs, outputs, batch_size ) {};
		~Sigmoid() {};
		
};

class ReLU : public CLLayer {

	public:
	
		void forward() {
		
			_TIMED_CALL_ ( cl_elementwise ( y, x, "relu" ) );
			
		}
		
		void backward() {
		
			_TIMED_CALL_ ( cl_elementwise ( dx, dy, y, "drelu" ) );
			
		}
		
		ReLU ( cl_ctx *cl_env, size_t inputs, size_t outputs, size_t batch_size ) : CLLayer ( cl_env, inputs, outputs, batch_size ) {};
		~ReLU() {};
		
};

class Softmax : public CLLayer {

	public:
	
		void forward() {
		
			_TIMED_CALL_ ( cl_sub_max_coeff ( x ) );
			_TIMED_CALL_ ( cl_softmax ( y, x ) );
			
			
		}
		
		void backward() {
		
			_TIMED_CALL_ ( cl_elementwise ( dx, dy, y, "dsoftmax" ) );
			
		}
		
		
		Softmax ( cl_ctx *cl_env, size_t inputs, size_t outputs, size_t batch_size ) : CLLayer ( cl_env, inputs, outputs, batch_size ) {};
		~Softmax() {};
		
};

#endif