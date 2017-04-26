/*
* @Author: kmrocki
* @Date:   2016-02-24 15:28:10
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-26 16:20:54
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
	cl_matrix<float> batch, targets, errors;
	cl_matrix<int> random_ints;

	const size_t batch_size;
	const size_t inputs;
	const size_t outputs;

	float smooth_loss = -1.0f;

	cl_ctx& ctx;

	void forward(cl_matrix<float>& input_data) {

		//copy inputs ptr to the lowest point in the network
		layers[0]->x.device_data = input_data.device_data;

		//compute forward activations
		for (size_t i = 0; i < layers.size(); i++) {

			//link inputs-outputs
			if (i > 0) layers[i]->x.device_data = layers[i - 1]->y.device_data;

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

	void backward(cl_matrix<float>& t) {

		//set targets ptr at the top
		layers.back()->dy.device_data = t.device_data;

		//propagate error backward
		for (int i = layers.size() - 1; i >= 0; i--) {

			layers[i]->resetGrads();
			layers[i]->backward();

			//dy(previous layer) = dx(current layer)
			if (i > 0)  layers[i - 1]->dy.device_data = layers[i]->dx.device_data;


		}

	}

	void update(float alpha) {

		//update all layers according to gradients
		for (size_t i = 0; i < layers.size(); i++) {

			layers[i]->applyGrads(alpha);

		}

	}

	void train(datapoints& data, float alpha, size_t iterations, bool show_loss = false) {

		for (size_t ii = 0; ii < iterations; ii++) {

			matrix_randi_host(random_ints.ref_host_data, 0, data.x.cols() - 1);
			random_ints.sync_device();

			// make batch
			cl_gather_data ( data.x, batch, random_ints );

			forward(batch);

			if (show_loss) {

				layers.back()->y.sync_host();
				targets.sync_host();

				double loss = cross_entropy_host(errors.ref_host_data, layers.back()->y.ref_host_data, targets.ref_host_data);
				smooth_loss = isNaNInf(loss) ? smooth_loss : (smooth_loss > 0 ? smooth_loss * 0.99f + loss * 0.01f : loss);

				if (ii % 100 == 99) std::cout << "[" << ii + 1 << "/" << iterations << "] Loss = " << smooth_loss << std::endl;
			}

			// make batch labels
			cl_gather_data ( data.y1, targets, random_ints );

			//backprogagation
			backward(targets);

			//apply changes
			update(alpha);

		}

	}


	void test(datapoints& data) {

		size_t correct = 0;

		for (size_t ii = 0; ii < data.x.cols(); ii += batch_size) {

			linspace_host(random_ints.ref_host_data, ii, ii + batch_size);
			random_ints.sync_device();

			// make batch
			cl_gather_data ( data.x, batch, random_ints );

			forward(batch);

			// make batch labels
			cl_gather_data ( data.y1, targets, random_ints );

			layers.back()->y.sync_host();
			targets.sync_host();

			correct += count_correct_predictions_host(layers.back()->y.ref_host_data, targets.ref_host_data);

		}

		std::cout << "Test % correct = " << 100.0 * (double)correct / (double)(data.x.cols()) << std::endl;

	}

	CLNN(cl_ctx& _ctx, size_t _batch_size, size_t _inputs, size_t _outputs) :
		batch_size(_batch_size), inputs(_inputs), outputs(_outputs), ctx(_ctx) {

		batch = cl_matrix<float>(&ctx, {inputs, batch_size});
		targets = cl_matrix<float>(&ctx, {outputs, batch_size});
		errors = cl_matrix<float>(&ctx, {outputs, batch_size});
		random_ints = cl_matrix<int>(&ctx, {batch_size, 1});

	}

	~CLNN() {

		for (size_t i = 0; i < layers.size(); i++) {

			delete(layers[i]);

		}

	}
};

#endif