/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-04-24 16:14:23
* @Last Modified by:   Kamil Rocki
* @Last Modified time: 2017-04-25 18:21:57
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

void nntest();

int main ( void ) {

	nntest();
	
}

void nntest() {

	size_t epochs = 1000;
	size_t batch_size = 250;
	double learning_rate = 1e-4f;
	
	cl_ctx ocl;
	int requested_cl_device = 1;
	if ( ocl.init ( requested_cl_device ) != 0 ) printf ( "opencl init failed!\n" );
	ocl.asynchronous = true;
	
	CLNN nn ( ocl, batch_size );
	
	nn.layers.push_back ( new Linear ( &ocl, 28 * 28, 400, batch_size ) );
	nn.layers.push_back ( new ReLU ( &ocl, 400, 400, batch_size ) );
	nn.layers.push_back ( new Linear ( &ocl, 400, 400, batch_size ) );
	nn.layers.push_back ( new ReLU ( &ocl, 400, 400, batch_size ) );
	nn.layers.push_back ( new Linear ( &ocl, 400, 256, batch_size ) );
	nn.layers.push_back ( new ReLU ( &ocl, 256, 256, batch_size ) );
	nn.layers.push_back ( new Linear ( &ocl, 256, 100, batch_size ) );
	nn.layers.push_back ( new ReLU ( &ocl, 100, 100, batch_size ) );
	nn.layers.push_back ( new Linear ( &ocl, 100, 10, batch_size ) );
	nn.layers.push_back ( new Softmax ( &ocl, 10, 10, batch_size ) );
	
	//[60000, 784]
	std::deque<datapoint> train_data = MNISTImporter::importFromFile ( "data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte" );
	//[10000, 784]
	std::deque<datapoint> test_data = MNISTImporter::importFromFile ( "data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte" );
	
	std::cout << "CL mem allocated: " << ( double ) cl_mem_allocated / ( double ) ( 1 << 20 ) << " MB\n";
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
	
	for ( size_t e = 0; e < epochs; e++ ) {
	
		std::cout << std::endl << "Epoch " << e + 1 << std::endl;
		start = std::chrono::system_clock::now();
		nn.train ( train_data, learning_rate, 1000 );
		end = std::chrono::system_clock::now();
		double difference = ( double ) std::chrono::duration_cast<std::chrono::microseconds> ( end - start ).count() / ( double ) 1e6;
		std::cout << "T = " << difference << " s" << std::endl;
		nn.test ( test_data );
		
	}
}