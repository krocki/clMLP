/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-04-24 16:14:23
* @Last Modified by:   Kamil Rocki
* @Last Modified time: 2017-04-27 10:04:26
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

void nntest ( int device, cl_device_type dev_type = CL_DEVICE_TYPE_ALL );


int main ( int argc, char **argv ) {

	int requested_cl_device = 0;
	
	if ( argc > 1 )
		requested_cl_device = atoi ( argv[1] );
		
	nntest ( requested_cl_device, CL_DEVICE_TYPE_GPU );
	
}

void nntest ( int device, cl_device_type dev_type ) {

	size_t epochs = 1000;
	size_t batch_size = 100;
	float learning_rate = 1e-4f;
	bool show_training_loss = true;
	
	cl_ctx ocl;
	
	if ( ocl.init ( device, dev_type ) != 0 ) printf ( "opencl init failed!\n" );
	
	CLNN nn ( ocl, batch_size, 28 * 28, 10 );
	
	nn.layers.push_back ( new Linear ( &ocl, 28 * 28, 400, batch_size ) );
	nn.layers.push_back ( new ReLU ( &ocl, 400, 400, batch_size ) );
	nn.layers.push_back ( new Linear ( &ocl, 400, 400, batch_size ) );
	nn.layers.push_back ( new ReLU ( &ocl, 400, 400, batch_size ) );
	nn.layers.push_back ( new Linear ( &ocl, 400, 10, batch_size ) );
	nn.layers.push_back ( new Softmax ( &ocl, 10, 10, batch_size ) );
	
	datapoints train_data = MNISTImporter::importFromFile ( ocl,  "data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", 60000 );
	datapoints test_data = MNISTImporter::importFromFile ( ocl, "data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", 10000 );
	
	std::cout << "CL mem allocated: " << ( double ) cl_mem_allocated / ( double ) ( 1 << 20 ) << " MB\n";
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
	
	for ( size_t e = 0; e < epochs; e++ ) {
	
		std::cout << std::endl << "Epoch " << e + 1 << std::endl;
		start = std::chrono::system_clock::now();
		nn.train ( train_data, learning_rate, 1000, show_training_loss );
		end = std::chrono::system_clock::now();
		double difference = ( double ) std::chrono::duration_cast<std::chrono::microseconds> ( end - start ).count() / ( double ) 1e6;
		std::cout << "T = " << difference << " s, " << 1e-9 * ( ( long double ) cl_flops_performed / ( long double ) difference ) << " GF/s" << std::endl;
		cl_flops_performed = 0L;
		nn.test ( test_data );
		
	}
}
