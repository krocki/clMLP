/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-04-25 03:59:24
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-25 15:37:30
*/

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <clBLAS.h>
#include <containers/dict.h>

#ifndef __CL_CTX_H__
#define __CL_CTX_H__

class cl_ctx {

  private:

	cl_platform_id platform = 0;
	cl_device_id device = 0;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	cl_context _ctx = 0;
	cl_command_queue _queue = 0;
	cl_program program = 0;

	int ret = 0;

	std::vector < deviceInfo > availableDevices;

  public:

	cl_int err;
	cl_event event = NULL;

	Dict<cl_kernel> kernels1; // unary
	Dict<cl_kernel> kernels2; // binary
	Dict<cl_kernel> kernels3; // ternary

	Dict<cl_kernel> kernels_mat_scalar;
	Dict<cl_kernel> kernels_colwise;

	size_t local_work_size = 1;

	int init(int requested_device = 0) {

		printf("Querying OpenCL...\n");

		availableDevices.clear();

		std::vector < deviceInfo > clDevices = clUtils::listDevices();
		availableDevices.insert (availableDevices.end(), clDevices.begin(), clDevices.end() );

		for (unsigned int i = 0; i < availableDevices.size(); i++) {
			printf ("[%2d]: %s [%s, local id = %ld]\n", i,
			        availableDevices[i].name.c_str(), availableDevices[i].type.c_str(),
			        availableDevices[i].localNum);
		}

		int requestedDevice = requested_device;
		unsigned selectedDevice = 0;

		if (requestedDevice >= 0 && requestedDevice < (int)availableDevices.size())
			selectedDevice = (unsigned)requestedDevice;
		else
			printf ("Device %d is not available!\n", requestedDevice);

		printf("Selected Device: %u (%s) \n", selectedDevice, availableDevices[selectedDevice].name.c_str());

		/* Setup OpenCL environment. */
		err = clGetPlatformIDs(1, &platform, NULL);
		if (err != CL_SUCCESS) {
			printf( "clGetPlatformIDs() failed with %d\n", err );
			return 1;
		}

		// err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
		// if (err != CL_SUCCESS) {
		//     printf( "clGetDeviceIDs() failed with %d\n", err );
		//     return 1;
		// }

		device = (cl_device_id)availableDevices[selectedDevice].localNum;

		cl_dev_info dev_properties = clUtils::getDevice ( device );

		printf("device_string: %s\n", dev_properties.device_string.c_str());
		printf("compute_units: %u\n", dev_properties.compute_units);
		printf("workgroup_size: %zu\n", dev_properties.workgroup_size);
		printf("global_mem_size: %lu\n", dev_properties.global_mem_size);
		printf("local_mem_size: %lu\n", dev_properties.local_mem_size);
		printf("preferred_vector: %u\n", dev_properties.preferred_vector);

		local_work_size = dev_properties.workgroup_size;

		//props[1] = (cl_context_properties)platform;
		_ctx = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
		if (err != CL_SUCCESS) {
			printf( "clCreateContext() failed with %d\n", err );
			return 1;
		}

		_queue = clCreateCommandQueue(_ctx, device, 0, &err);
		if (err != CL_SUCCESS) {
			printf( "clCreateCommandQueue() failed with %d\n", err );
			clReleaseContext(_ctx);
			return 1;
		}

		/* Setup clblas. */
		err = clblasSetup();

		if (err != CL_SUCCESS) {
			printf("clblasSetup() failed with %d\n", err);
			clReleaseCommandQueue(_queue);
			clReleaseContext(_ctx);
			return 1;
		}

		// compiling programs
		printf("compiling programs\n");

		program = clUtils::compileProgram ("./src/opencl/kernels/elementwise_ops.cl", _ctx, device);

		if (!program) {
			printf("compileProgram() failed.");
			clReleaseCommandQueue(_queue);
			clReleaseContext(_ctx);
			return 1;
		}

		kernels1["logistic"] = clCreateKernel (program, "logistic1", &err);
		kernels1["relu"] = clCreateKernel (program, "rectify1", &err);
		kernels1["exp"] = clCreateKernel (program, "expf1", &err);
		kernels2["logistic"] = clCreateKernel (program, "logistic2", &err);
		kernels2["relu"] = clCreateKernel (program, "rectify2", &err);
		kernels2["colsumdiv"] = clCreateKernel (program, "colsumdiv", &err);
		kernels2["exp"] = clCreateKernel (program, "expf2", &err);
		kernels2["sub"] = clCreateKernel (program, "sub2", &err);
		kernels3["sub"] = clCreateKernel (program, "sub3", &err);
		kernels3["drelu"] = clCreateKernel (program, "drelu", &err);
		kernels3["dlogistic"] = clCreateKernel (program, "dlogistic", &err);
		kernels3["dsoftmax"] = clCreateKernel (program, "dsoftmax", &err);
		kernels3["fmad"] = clCreateKernel (program, "fmad", &err);
		kernels_mat_scalar["sub"] = clCreateKernel (program, "sub1", &err);



		if (err != CL_SUCCESS) {
			printf("clCreateKernel() failed with %d\n", err);
			clReleaseCommandQueue(_queue);
			clReleaseContext(_ctx);
			return 1;
		}

		return 0;
	}

	~cl_ctx() {

		/* Finalize work with clblas. */
		clblasTeardown();

		for (size_t i = 0; i < kernels1.matrices.size(); i++)
			clReleaseKernel (kernels1.matrices[i]);
		for (size_t i = 0; i < kernels2.matrices.size(); i++)
			clReleaseKernel (kernels2.matrices[i]);
		for (size_t i = 0; i < kernels3.matrices.size(); i++)
			clReleaseKernel (kernels3.matrices[i]);
		for (size_t i = 0; i < kernels_mat_scalar.matrices.size(); i++)
			clReleaseKernel (kernels_mat_scalar.matrices[i]);
		for (size_t i = 0; i < kernels_colwise.matrices.size(); i++)
			clReleaseKernel (kernels_colwise.matrices[i]);

		clReleaseProgram (program);

		/* Release OpenCL working objects. */
		clReleaseCommandQueue(_queue);
		clReleaseContext(_ctx);

	}

	cl_context& ctx() { return _ctx; }
	cl_command_queue& queue() { return _queue; }

};

#endif
