// #pragma OPENCL EXTENSION cl_amd_printf : enable

typedef struct device_nn_layer {

	// public:

	float* input;
	float* output;

} device_nn_layer;

__kernel void forward (__global device_nn_layer* layers, const unsigned int count) {
	int i = get_global_id (0);

	// 1. sgemm(y0, W0, x0);

	// 2. relu(y1, y0);

	// 3. sgemm(y2, W2, y1);

	// 4. relu(y3, y2);

	// 5. sgemm(y4, W4, y3);

	// 6. relu(y5, y4);

	// 7. sgemm(y6, W6, y5);

	// 8. softmax(y7, y6);

	// nn.layers.push_back (new Linear (&ocl, 28 * 28, 400, batch_size) );
	// nn.layers.push_back (new ReLU (&ocl, 400, 400, batch_size) );
	// nn.layers.push_back (new Linear (&ocl, 400, 256, batch_size) );
	// nn.layers.push_back (new ReLU (&ocl, 256, 256, batch_size) );
	// nn.layers.push_back (new Linear (&ocl, 256, 100, batch_size) );
	// nn.layers.push_back (new ReLU (&ocl, 100, 100, batch_size) );
	// nn.layers.push_back (new Linear (&ocl, 100, 10, batch_size) );
	// nn.layers.push_back (new Softmax (&ocl, 10, 10, batch_size) );

}
