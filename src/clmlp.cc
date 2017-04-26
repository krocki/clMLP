/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-04-24 16:14:23
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-26 16:12:27
*/

#include <opencl/cl_utils.h>
#include <opencl/cl_defs.h>
#include <opencl/cl_ctx.h>
#include <opencl/cl_matrix.h>

#include <utils.h>
#include <io/import.h>
#include <nn/clnn_utils.h>
#include <nn/cllayers.h>
#include <nn/clnn.h>

#include <vector>

void nntest(int device);


int main ( int argc, char **argv ) {

	int requested_cl_device = 1;

	if (argc > 1)
		requested_cl_device = atoi(argv[1]);

	nntest(requested_cl_device);

}

void nntest(int device) {

	size_t epochs = 1000;
	size_t batch_size = 250;
	float learning_rate = 1e-3f;

	cl_ctx ocl;

	if ( ocl.init ( device ) != 0 ) printf ( "opencl init failed!\n" );

	CLNN nn ( ocl, batch_size, 28 * 28, 10 );

	nn.layers.push_back ( new Linear ( &ocl, 28 * 28, 256, batch_size ) );
	nn.layers.push_back ( new ReLU ( &ocl, 256, 256, batch_size ) );
	nn.layers.push_back ( new Linear ( &ocl, 256, 100, batch_size ) );
	nn.layers.push_back ( new ReLU ( &ocl, 100, 100, batch_size ) );
	nn.layers.push_back ( new Linear ( &ocl, 100, 10, batch_size ) );
	nn.layers.push_back ( new Softmax ( &ocl, 10, 10, batch_size ) );

	datapoints train_data = MNISTImporter::importFromFile (ocl,  "data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", 60000 );
	datapoints test_data = MNISTImporter::importFromFile (ocl, "data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", 10000 );

	std::cout << "CL mem allocated: " << ( double ) cl_mem_allocated / ( double ) ( 1 << 20 ) << " MB\n";

	std::chrono::time_point<std::chrono::system_clock> start, end;

	for ( size_t e = 0; e < epochs; e++ ) {

		std::cout << std::endl << "Epoch " << e + 1 << std::endl;
		start = std::chrono::system_clock::now();
		nn.train ( train_data, learning_rate, 1000 );
		end = std::chrono::system_clock::now();
		double difference = ( double ) std::chrono::duration_cast<std::chrono::microseconds> ( end - start ).count() / ( double ) 1e6;
		std::cout << "T = " << difference << " s, " << 1e-9 * ((long double)cl_flops_performed / (long double)difference) << " GF/s" << std::endl;
		cl_flops_performed = 0L;
		nn.test ( test_data );

	}
}