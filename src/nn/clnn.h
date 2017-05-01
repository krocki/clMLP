/*
    @Author: kmrocki
    @Date:   2016-02-24 15:28:10
    @Last Modified by:   Kamil Rocki
    @Last Modified time: 2017-04-28 19:51:47
*/

#ifndef __CLNN_H__
#define __CLNN_H__

#include <deque>
#include <opencl/cl_ctx.h>
#include <opencl/cl_matrix.h>
#include <nn/cllayers.h>

class CLNN {

  public:

	std::deque<CLLayer*> layers;
	cl_matrix<float> batch, targets, errors, logprobs;

	cl_matrix<int> rands_int;
	cl_matrix<float> rands_uniform, rands_normal;

	const size_t batch_size;
	const size_t inputs;
	const size_t outputs;

	float smooth_loss = -1.0f;

	cl_ctx& ctx;

	void forward (cl_matrix<float>& input_data) {
		//copy inputs ptr to the lowest point in the network
		layers[0]->x.ref_device_data = input_data.device_data;

		//compute forward activations
		for (size_t i = 0; i < layers.size(); i++) {
			//link inputs-outputs
			if (i > 0) layers[i]->x.ref_device_data = layers[i - 1]->y.device_data;

			layers[i]->forward();
		}
	}

	void sync_params_device() {
		for (size_t i = 0; i < layers.size(); i++)
			layers[i]->sync_device();
	}

	void sync_params_host() {
		for (size_t i = 0; i < layers.size(); i++)
			layers[i]->sync_host();
	}

	void backward (cl_matrix<float>& t) {
		//set targets ptr at the top
		layers.back()->dy.ref_device_data = t.device_data;

		//propagate error backward
		for (int i = layers.size() - 1; i >= 0; i--) {
			layers[i]->resetGrads();
			layers[i]->backward();

			//dy(previous layer) = dx(current layer)
			if (i > 0)
				layers[i - 1]->dy.ref_device_data = layers[i]->dx.device_data;
		}
	}

	void update (float alpha) {
		//update all layers according to gradients
		for (size_t i = 0; i < layers.size(); i++)
			layers[i]->applyGrads (alpha);
	}

	void train (datapoints& data, float alpha, size_t iterations, bool show_loss = false) {

		for (size_t ii = 0; ii < iterations; ii++) {

			// generate numbers {0, . . . , data.x.cols() - 1}
			//TODO: batch is always the same, because rands_int are always the same
			cl_matrix_randi (rands_int, 0, data.x.cols() - 1, false);
			//rands_int.sync_host();
			//std::cout << "int rands" << std::endl;
			//std::cout << rands_int.ref_host_data << std::endl;
			// make batch
			cl_gather_data (data.x, batch, rands_int);
			forward (batch);
			// make batch labels
			cl_gather_data (data.y1, targets, rands_int);

			if (show_loss) {

				layers.back()->y.sync_host();
				targets.sync_host();
				double loss;
				loss = cross_entropy (errors, logprobs, layers.back()->y, targets);
				smooth_loss = isNaNInf (loss) ? smooth_loss : (smooth_loss > 0 ? smooth_loss * 0.99f + loss * 0.01f : loss);

				if (ii % 100 == 99) std::cout << "[" << ii + 1 << "/" << iterations << "] Loss = " << smooth_loss << std::endl;
			}

			//backprogagation
			backward (targets);
			//apply changes
			update (alpha);
		}
	}


	void test (datapoints& data) {
		size_t correct = 0;

		for (size_t ii = 0; ii < data.x.cols(); ii += batch_size) {
			linspace_host (rands_int.ref_host_data, ii, ii + batch_size);
			rands_int.sync_device();
			//std::cout << "int rands" << std::endl;
			//std::cout << rands_int.ref_host_data << std::endl;
			// make batch
			cl_gather_data (data.x, batch, rands_int);
			forward (batch);
			// make batch labels
			cl_gather_data (data.y1, targets, rands_int);
			layers.back()->y.sync_host();
			targets.sync_host();
			correct += count_correct_predictions_host (layers.back()->y.ref_host_data, targets.ref_host_data);
		}

		std::cout << "Test % correct = " << 100.0 * (double) correct / (double) (data.x.cols() ) << std::endl;
	}

	CLNN (cl_ctx& _ctx, size_t _batch_size, size_t _inputs, size_t _outputs) :
		batch_size (_batch_size), inputs (_inputs), outputs (_outputs), ctx (_ctx) {
		batch = cl_matrix<float> (&ctx, {inputs, batch_size});
		targets = cl_matrix<float> (&ctx, {outputs, batch_size});
		errors = cl_matrix<float> (&ctx, {outputs, batch_size});
		logprobs = cl_matrix<float> (&ctx, {outputs, batch_size});
		rands_int = cl_matrix<int> (&ctx, {batch_size, 1});
		rands_uniform = cl_matrix<float> (&ctx, {512, 16});
		rands_normal = cl_matrix<float> (&ctx, {512, 16});
		rands_int = cl_matrix<int> (&ctx, {batch_size, 1});

		// test_rands();
	}

	void test_rands() {

		// test u01
		cl_matrix_rand ( rands_uniform, false );
		rands_uniform.sync_host();

		std::cout << "uniform rands" << std::endl;
		std::cout << rands_uniform.ref_host_data << std::endl;
		std::cout << "rands end" << std::endl;

		// test normal
		std::cout << "test normal" << std::endl;
		cl_matrix_randn ( rands_normal, false );
		rands_normal.sync_host();

		std::cout << "normal rands" << std::endl;
		std::cout << rands_normal.ref_host_data << std::endl;
		std::cout << "normalrands end" << std::endl;

		// test int uniform
		std::cout << "test int uniform" << std::endl;
		// generate numbers {0, . . . , 9}
		cl_matrix_randi ( rands_int, 0, 9, false );
		rands_int.sync_host();

		// make a histogram and count

		std::cout << "int rands" << std::endl;
		std::cout << rands_int.ref_host_data << std::endl;
		std::cout << "intrands end" << std::endl;

	}

	~CLNN() {
		for (size_t i = 0; i < layers.size(); i++)
			delete (layers[i]);
	}
};

#endif