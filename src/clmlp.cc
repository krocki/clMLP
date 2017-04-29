/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-04-24 16:14:23
* @Last Modified by:   Kamil Rocki
* @Last Modified time: 2017-04-28 19:51:47
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

void nntest ( int device, cl_device_type dev_type = CL_DEVICE_TYPE_ALL, int enable_profiling = 0 );

int main ( int argc, char **argv ) {

	int requested_cl_device = 1;
	int enable_profiling = 0;
	
	if ( argc > 1 ) requested_cl_device = atoi ( argv[1] );
	if ( argc > 2 ) enable_profiling = atoi ( argv[2] );
	
	nntest ( requested_cl_device, CL_DEVICE_TYPE_GPU, enable_profiling );
	
}

void nntest ( int device, cl_device_type dev_type, int enable_profiling ) {

	size_t epochs = 1000;
	size_t batch_size = 250;
	float learning_rate = 1e-4f;
	bool show_training_loss = true;
	
	start = std::chrono::system_clock::now();
	
	cl_ctx ocl ( enable_profiling, pdata );
	
	if ( ocl.init ( device, dev_type ) != 0 ) printf ( "opencl init failed!\n" );
	
	CLNN nn ( ocl, batch_size, 28 * 28, 10 );
	
	// ~98.1% after 60 epochs
	nn.layers.push_back ( new Linear ( &ocl, 28 * 28, 400, batch_size ) );
	nn.layers.push_back ( new ReLU ( &ocl, 400, 400, batch_size ) );
	nn.layers.push_back ( new Linear ( &ocl, 400, 256, batch_size ) );
	nn.layers.push_back ( new ReLU ( &ocl, 256, 256, batch_size ) );
	nn.layers.push_back ( new Linear ( &ocl, 256, 100, batch_size ) );
	nn.layers.push_back ( new ReLU ( &ocl, 100, 100, batch_size ) );
	nn.layers.push_back ( new Linear ( &ocl, 100, 10, batch_size ) );
	nn.layers.push_back ( new Softmax ( &ocl, 10, 10, batch_size ) );
	
	datapoints train_data = MNISTImporter::importFromFile ( ocl,  "data/mnist/train-images-idx3-ubyte", "data/mnist/train-labels-idx1-ubyte", 60000 );
	datapoints test_data = MNISTImporter::importFromFile ( ocl, "data/mnist/t10k-images-idx3-ubyte", "data/mnist/t10k-labels-idx1-ubyte", 10000 );
	
	std::cout << "CL mem allocated: " << ( double ) cl_mem_allocated / ( double ) ( 1 << 20 ) << " MB\n";
	
	std::chrono::time_point<std::chrono::system_clock> start, end;
	
	for ( size_t e = 0; e < epochs; e++ ) {
	
		std::cout << std::endl << "Epoch " << e + 1 << std::endl;
		
		_TIMED_CALL_ ( nn.train ( train_data, learning_rate, 1000, show_training_loss ) );
		
		show_profiling_data ( pdata, SORT_BY_TIME_DESC, enable_profiling );
		
		_TIMED_CALL_ ( nn.test ( test_data ) );
		
	}
}
