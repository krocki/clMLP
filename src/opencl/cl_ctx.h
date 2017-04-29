/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-04-25 03:59:24
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-29 11:23:54
*/

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#ifdef CLTUNE
#include <cltune.h>
#endif

#include <opencl/cl_blas_defs.h>
#include <opencl/cl_rand.h>
#include <opencl/cl_prof.h>
#include <containers/dict.h>

#ifndef __CL_CTX_H__
#define __CL_CTX_H__

class cl_ctx {

  private:

	cl_platform_id platform = 0;
	cl_device_id device = 0;
	// cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	cl_context _ctx = 0;
	cl_command_queue _queue = 0;
	cl_program program_elementwise = 0;
	cl_program program_rand = 0;

	std::vector <deviceInfo> availableDevices;

  public:

	bool asynchronous = true;
	bool profiling_enabled = true;
	bool ooo_exec_enabled = false;

	cl_int err;
	cl_event event = NULL;

	// for profiling
	Dict<cl_event> cl_events;

	Dict<cl_kernel> kernels1; // unary
	Dict<cl_kernel> kernels2; // binary
	Dict<cl_kernel> kernels3; // ternary

	Dict<cl_kernel> kernels4; // etc

	Dict<cl_kernel> kernels_mat_scalar;
	Dict<cl_kernel> kernels_colwise;

	// from rand.cl
	Dict<cl_kernel> kernels_rand;
	Dict<prof_data> &pdata;

	bool get_workgroup_size_from_device = false;
	size_t local_work_size = 64;

	cl_ctx ( bool enable_profiling, Dict<prof_data> &_pdata ) : profiling_enabled ( enable_profiling ), pdata ( _pdata ) { }

	void get_profiling_data ( std::string key ) {

		clWaitForEvents ( 1, &cl_events[key] );

		if ( profiling_enabled ) {

			cl_ulong time_start, time_end;
			double total_time;

			clGetEventProfilingInfo ( cl_events[key], CL_PROFILING_COMMAND_START, sizeof ( time_start ), &time_start, NULL );
			clGetEventProfilingInfo ( cl_events[key], CL_PROFILING_COMMAND_END, sizeof ( time_end ), &time_end, NULL );
			total_time = time_end - time_start;

			pdata[key].time += total_time;
			pdata[key].key = key;
		}

	}

	// TODO: requested_device_type = GPU/CPU, ...
	int init ( int requested_device = 0, cl_device_type dev_type = CL_DEVICE_TYPE_ALL ) {

		printf ( "Querying OpenCL...\n" );

		availableDevices.clear();

		std::vector <deviceInfo> clDevices = clUtils::listDevices ( dev_type );
		availableDevices.insert ( availableDevices.end(), clDevices.begin(), clDevices.end() );

		for ( unsigned int i = 0; i < availableDevices.size(); i++ ) {
			printf ( "[%2d]: %s [%s, local id = %ld]\n", i,
			         availableDevices[i].name.c_str(), availableDevices[i].type.c_str(),
			         availableDevices[i].localNum );
		}

		int requestedDevice = requested_device;
		unsigned selectedDevice = 0;

		if ( requestedDevice >= 0 && requestedDevice < ( int ) availableDevices.size() )
			selectedDevice = ( unsigned ) requestedDevice;
		else
			printf ( "Device %d is not available!\n", requestedDevice );

		printf ( "Selected Device: %u (%s) \n", selectedDevice, availableDevices[selectedDevice].name.c_str() );

		/* Setup OpenCL environment. */
		err = clGetPlatformIDs ( 1, &platform, NULL );
		if ( err != CL_SUCCESS ) {
			printf ( "clGetPlatformIDs() failed with %d\n", err );
			return 1;
		}

		device = ( cl_device_id ) availableDevices[selectedDevice].localNum;

		cl_dev_info dev_properties = clUtils::getDevice ( device );

		printf ( "device_string: %s\n", dev_properties.device_string.c_str() );
		printf ( "compute_units: %u\n", dev_properties.compute_units );
		printf ( "workgroup_size: %u\n", dev_properties.workgroup_size );
		printf ( "global_mem_size: %llu\n", ( long long unsigned int ) dev_properties.global_mem_size );
		printf ( "local_mem_size: %llu\n", ( long long unsigned int ) dev_properties.local_mem_size );
		printf ( "preferred_vector: %u\n", dev_properties.preferred_vector );

		if ( get_workgroup_size_from_device )
			local_work_size = dev_properties.workgroup_size;

		//props[1] = (cl_context_properties)platform;

		_ctx = clCreateContext ( NULL, 1, &device, NULL, NULL, &err );
		if ( err != CL_SUCCESS ) {
			printf ( "clCreateContext() failed with %d\n", err );
			return 1;
		}

		/* create command queue */
		cl_command_queue_properties queue_properties = 0;
		if ( profiling_enabled ) queue_properties |= CL_QUEUE_PROFILING_ENABLE;
		if ( ooo_exec_enabled ) queue_properties |= CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
		const std::string queue_color_message = "\x1b[33m[ cpu profiling_enabled = " + std::to_string ( CPU_PROF_ENABLED ) + " ]\x1b[0m\n" + "\x1b[33m[ cl profiling_enabled = " + std::to_string ( CL_PROF_ENABLED ) + " ]\x1b[0m\n" + "\x1b[33m[ ooo_exec_enabled = " + std::to_string ( ooo_exec_enabled ) + " ]\x1b[0m\n";

		std::cout << std::endl << queue_color_message << std::endl;

		_queue = clCreateCommandQueue ( _ctx, device, queue_properties, &err );

		if ( err != CL_SUCCESS ) {
			printf ( "clCreateCommandQueue() failed with %d\n", err );
			clReleaseContext ( _ctx );
			return 1;
		}

		std::cout << "CL_BLAS_IMPL: " << CL_BLAS_IMPL << std::endl;

		/* Setup clblas. */
		err = CL_BLAS_INIT();
		/* Setup clRNG. */
		init_clrng ( _ctx, local_work_size );

		if ( err != CL_SUCCESS ) {
			printf ( "clblasSetup() failed with %d\n", err );
			clReleaseCommandQueue ( _queue );
			clReleaseContext ( _ctx );
			return 1;
		}
		// compiling programs
		printf ( "compiling programs\n" );

		program_elementwise = clUtils::compileProgram ( "./src/opencl/kernels/elementwise_ops.cl", _ctx, device, "" );

		if ( !program_elementwise ) {

			printf ( "program_elementwise compilation failed." );
			clReleaseCommandQueue ( _queue );
			clReleaseContext ( _ctx );
			return 1;
		}

		const char *clrng_build_flags = "-I/usr/local/include";

		program_rand = clUtils::compileProgram ( "./src/opencl/kernels/rand.cl", _ctx, device, clrng_build_flags );

		if ( !program_rand ) {

			printf ( "program_rand compilation failed." );
			clReleaseCommandQueue ( _queue );
			clReleaseContext ( _ctx );
			return 1;
		}

		kernels1["logistic"] = clCreateKernel ( program_elementwise, "logistic1", &err );
		kernels1["relu"] = clCreateKernel ( program_elementwise, "rectify1", &err );
		kernels1["exp"] = clCreateKernel ( program_elementwise, "expf1", &err );

		kernels2["logistic"] = clCreateKernel ( program_elementwise, "logistic2", &err );
		kernels2["relu"] = clCreateKernel ( program_elementwise, "rectify2", &err );
		kernels2["colsumdiv"] = clCreateKernel ( program_elementwise, "colsumdiv", &err );
		kernels2["exp"] = clCreateKernel ( program_elementwise, "expf2", &err );
		kernels2["sub"] = clCreateKernel ( program_elementwise, "sub2", &err );

		kernels3["sub"] = clCreateKernel ( program_elementwise, "sub3", &err );
		kernels3["drelu"] = clCreateKernel ( program_elementwise, "drelu", &err );
		kernels3["dlogistic"] = clCreateKernel ( program_elementwise, "dlogistic", &err );
		kernels3["dsoftmax"] = clCreateKernel ( program_elementwise, "dsoftmax", &err );
		kernels3["fmad"] = clCreateKernel ( program_elementwise, "fmad", &err );
		kernels3["fmad_lmem"] = clCreateKernel ( program_elementwise, "fmad_lmem", &err );

		kernels4["gather_data"] = clCreateKernel ( program_elementwise, "gather_data", &err );

		kernels_mat_scalar["sub"] = clCreateKernel ( program_elementwise, "sub1", &err );

		kernels_rand["uniform01"] = clCreateKernel ( program_rand, "uniform01", &err );
		kernels_rand["randi"] = clCreateKernel ( program_rand, "randi", &err );
		kernels_rand["normal"] = clCreateKernel ( program_rand, "normal", &err );

		if ( err != CL_SUCCESS ) {
			printf ( "clCreateKernel() failed with %d\n", err );
			clReleaseCommandQueue ( _queue );
			clReleaseContext ( _ctx );
			return 1;
		}

		const std::string color_message = "\x1b[33m[ workgroup_size = " + std::to_string ( local_work_size ) + " ]\x1b[0m\n";

		std::cout << std::endl << color_message << std::endl;

#ifdef CLTUNE
		// Vector dimension
		const auto kVectorSize = size_t{256 * 1024};

		// Creates the vectors and fills them with some example data
		std::vector<float> vec_a ( kVectorSize, 1.0f );

		// Initializes the tuner (platform 0, device 0)
		cltune::Tuner tuner ( size_t{0}, static_cast<size_t> ( requested_device ) );

		// Adds the kernel. The total number of threads (the global size) is equal to 'kVectorSize', and
		// the base number of threads per work-group/thread-block (the local size) is 1. This number is
		// then multiplied by the 'GROUP_SIZE' parameter, which can take any of the specified values.
		const auto id = tuner.AddKernel ( {"./src/opencl/kernels/elementwise_ops.cl"}, "logistic1", {kVectorSize}, {1} );
		tuner.AddParameter ( id, "GROUP_SIZE", {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048} );
		tuner.MulLocalSize ( id, {"GROUP_SIZE"} );

		// Sets the function's arguments
		tuner.AddArgumentInput ( vec_a );
		tuner.AddArgumentScalar ( static_cast<int> ( kVectorSize ) );

		// Starts the tuner
		tuner.SetNumRuns ( 10 );
		tuner.Tune();

		// Prints the results to screen
		tuner.PrintToScreen();
#endif

		return 0;
	}

	~cl_ctx() {

		destroy_clrng();

		/* Finalize work with clblas. */
		CL_BLAS_TEARDOWN();

		for ( size_t i = 0; i < kernels1.entries.size(); i++ )
			clReleaseKernel ( kernels1.entries[i] );
		for ( size_t i = 0; i < kernels2.entries.size(); i++ )
			clReleaseKernel ( kernels2.entries[i] );
		for ( size_t i = 0; i < kernels3.entries.size(); i++ )
			clReleaseKernel ( kernels3.entries[i] );
		for ( size_t i = 0; i < kernels4.entries.size(); i++ )
			clReleaseKernel ( kernels4.entries[i] );
		for ( size_t i = 0; i < kernels_mat_scalar.entries.size(); i++ )
			clReleaseKernel ( kernels_mat_scalar.entries[i] );
		for ( size_t i = 0; i < kernels_colwise.entries.size(); i++ )
			clReleaseKernel ( kernels_colwise.entries[i] );
		for ( size_t i = 0; i < kernels_rand.entries.size(); i++ )
			clReleaseKernel ( kernels_rand.entries[i] );

		clReleaseProgram ( program_elementwise );
		clReleaseProgram ( program_rand );

		/* Release OpenCL working objects. */
		clReleaseCommandQueue ( _queue );
		clReleaseContext ( _ctx );

	}

	cl_context &ctx() { return _ctx; }
	cl_command_queue &queue() { return _queue; }

};

#endif