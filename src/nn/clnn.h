/*
* @Author: kmrocki
* @Date:   2016-02-24 15:28:10
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-25 15:51:23
*/

#ifndef __CLNN_H__
#define __CLNN_H__

#include <opencl/cl_ctx.h>
#include <nn/cllayers.h>

class CLNN {

  public:

	std::deque<CLLayer*> layers;
	Matrix batch, targets, encoding;
	Eigen::VectorXi random_numbers;

	const size_t batch_size;
	float smooth_loss = -1.0f;

	cl_ctx& ctx;

	void forward(Matrix& input_data) {

		//copy inputs to the lowest point in the network
		layers[0]->x.ref_host_data = input_data;
		layers[0]->x.sync_device();

		//compute forward activations
		for (size_t i = 0; i < layers.size(); i++) {

			if (i > 0) {
				layers[i]->x.device_data = layers[i - 1]->y.device_data;
			}

			layers[i]->forward();


			//x(next layer) = y(current layer)
			// if (i + 1 < layers.size()) {
			// 	layers[i + 1]->x.ref_host_data = layers[i]->y.ref_host_data;
			// 	//layers[i + 1]->x.device_data = layers[i]->y.device_data;
			// }
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

	void backward(Matrix& t) {

		//set targets at the top
		layers.back()->dy.ref_host_data = t;
		layers.back()->dy.sync_device();

		//propagate error backward
		for (int i = layers.size() - 1; i >= 0; i--) {

			layers[i]->resetGrads();
			layers[i]->backward();

			//dy(previous layer) = dx(current layer)
			if (i > 0) {

				layers[i - 1]->dy.device_data = layers[i]->dx.device_data;

				// layers[i - 1]->dy.ref_host_data = layers[i]->dx.ref_host_data;
			}

		}

	}

	void update(double alpha) {

		//update all layers according to gradients
		for (size_t i = 0; i < layers.size(); i++) {

			layers[i]->applyGrads(alpha);

		}

	}

	void train(std::deque<datapoint>& data, double alpha, size_t iterations) {

		//get random examples of size batch_size from data
		size_t classes = 10;
		random_numbers.resize ( batch_size );
		batch.resize ( data[0].x.rows(), batch_size );
		targets.resize ( classes, batch_size );
		encoding = Matrix::Identity ( classes, classes );

		for (size_t ii = 0; ii < iterations; ii++) {

			matrix_randi(random_numbers, 0, data.size() - 1);

			// [784 x batch_size]
			make_batch ( batch, data, random_numbers );
			make_targets ( targets, encoding, data, random_numbers );

			// sync_params_device();
			//forward activations
			forward(batch);

			layers.back()->y.sync_host();
			double loss = cross_entropy(layers.back()->y.ref_host_data, targets);
			smooth_loss = isNaNInf(loss) ? smooth_loss : (smooth_loss > 0 ? smooth_loss * 0.99f + loss * 0.01f : loss);

			if (ii % 100 == 99)
				std::cout << "[" << ii + 1 << "/" << iterations << "] Loss = " << smooth_loss << std::endl;

			//backprogagation
			backward(targets);

			//apply changes
			update(alpha);

		}

	}


	void test(std::deque<datapoint>& data) {

		Eigen::VectorXi numbers(batch_size);
		size_t classes = 10;
		size_t correct = 0;

		for (size_t ii = 0; ii < data.size(); ii += batch_size) {

			linspace(numbers, ii, ii + batch_size);

			batch.resize ( data[0].x.rows(), batch_size );
			targets.resize ( classes, batch_size );
			encoding = Matrix::Identity ( classes, classes );

			make_batch ( batch, data, numbers );
			make_targets ( targets, encoding, data, numbers );

			forward(batch);

			layers.back()->y.sync_host();
			correct += count_correct_predictions(layers.back()->y.ref_host_data, targets);

		}

		std::cout << "Test % correct = " << 100.0 * (double)correct / (double)(data.size()) << std::endl;

	}

	CLNN(cl_ctx& _ctx, size_t minibatch_size) :
		batch_size(minibatch_size), ctx(_ctx) {}

	~CLNN() {

		for (size_t i = 0; i < layers.size(); i++) {

			delete(layers[i]);

		}

	}
};

#endif