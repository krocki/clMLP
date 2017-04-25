/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-04-24 16:58:15
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-25 16:13:15
*/

#include <Eigen/Dense>

#ifndef __CL_MATRIX__
#define __CL_MATRIX__

int cl_alloc_bytes(cl_ctx* clctx, cl_mem& buffer, size_t bytes, cl_mem_flags flags);
int cl_alloc_from_matrix(cl_ctx* ctx, cl_mem& buffer, Eigen::MatrixXf& h);
int cl_copy_matrix_to_host(cl_ctx* ctx, Eigen::MatrixXf& dst, cl_mem device_data);
int cl_copy_matrix_to_device(cl_ctx* ctx, cl_mem device_data, Eigen::MatrixXf& src);

unsigned long cl_mem_allocated = 0L;

class cl_matrix {

  public:

	cl_mem device_data;

	//temp buffers
	cl_mem scratchBuf;

	cl_mem iMax; // index of max value (updated through cl_max_coeff(cl_matrix& m)) - device mem
	cl_uint indexMax = 0; // index of max value (updated through cl_max_coeff(cl_matrix& m))
	unsigned int lenScratchBuf = 0;

	Eigen::MatrixXf host_data;
	Eigen::MatrixXf& ref_host_data;
	cl_ctx* matrix_ctx;

	cl_matrix(cl_ctx* ctx = nullptr) : ref_host_data(host_data), matrix_ctx(ctx) {

		device_data = nullptr;
		scratchBuf = nullptr;
		iMax = nullptr;

	}

	cl_matrix(cl_ctx* ctx, Eigen::Vector2i size) : cl_matrix(ctx) {

		host_data = Eigen::MatrixXf(size[0], size[1]);
		host_data.setZero();
		ref_host_data = host_data;
		alloc_device_mem();

	};

	cl_matrix(cl_ctx* ctx, Eigen::MatrixXf& m) : cl_matrix(ctx) {

		host_data = m;
		ref_host_data = host_data;
		alloc_device_mem();

	};

	cl_matrix &operator= ( const cl_matrix &other ) {

		host_data = other.host_data;
		ref_host_data = host_data;
		matrix_ctx = other.matrix_ctx;
		free_device_mem();
		alloc_device_mem();

	};

	size_t rows() const { return ref_host_data.rows(); }
	size_t cols() const { return ref_host_data.cols(); }
	size_t length() const { return rows() * cols(); }

	Eigen::Vector2i size() const { return { rows(), cols() }; }

	cl_matrix ( const cl_matrix &other ) : cl_matrix(other.matrix_ctx, other.size()) {}

	int setZero() {

		float zero = 0.0f;
		// ref_host_data.setZero();

		matrix_ctx->err = clEnqueueFillBuffer(matrix_ctx->queue(), device_data, &zero, sizeof(float), 0, length() * sizeof(float), 0, nullptr, nullptr);

		if (matrix_ctx->err != CL_SUCCESS) {

			printf( "clEnqueueFillBuffer with %d - %s\n", matrix_ctx->err, oclErrorString(matrix_ctx->err));
			return 1;

		}

	}

	int setOnes() {

		float one = 1.0f;
		// ref_host_data.setOnes();

		matrix_ctx->err = clEnqueueFillBuffer(matrix_ctx->queue(), device_data, &one, sizeof(float), 0, length() * sizeof(float), 0, nullptr, nullptr);

		if (matrix_ctx->err != CL_SUCCESS) {

			printf( "clEnqueueFillBuffer with %d - %s\n", matrix_ctx->err, oclErrorString(matrix_ctx->err));
			return 1;

		}

		return 0;

	}

	~cl_matrix() {

		free_device_mem();

	};

	void alloc_device_mem() {

		cl_alloc_from_matrix(matrix_ctx, device_data, ref_host_data);

	}

	void free_device_mem() {

		if (device_data)
			clReleaseMemObject((cl_mem)device_data);
		if (scratchBuf)
			clReleaseMemObject((cl_mem)scratchBuf);
		if (iMax)
			clReleaseMemObject((cl_mem)iMax);

	}

	void sync_device() {

		cl_copy_matrix_to_device(matrix_ctx, device_data, ref_host_data);

	}

	void sync_host() {

		cl_copy_matrix_to_host(matrix_ctx, ref_host_data, device_data);

	}

};

int cl_alloc_bytes(cl_ctx* clctx, cl_mem& buffer, size_t bytes, cl_mem_flags flags = CL_MEM_READ_WRITE) {

	clCreateBuffer(clctx->ctx(), flags, bytes, NULL, &clctx->err);

	if (clctx->err != CL_SUCCESS) {

		printf( "clCreateBuffer failed with %d - %s\n", clctx->err, oclErrorString(clctx->err));
		return 1;

	}

	return 0;
}

int cl_alloc_from_matrix(cl_ctx* clctx, cl_mem& buffer, Eigen::MatrixXf& h) {

	size_t alloc_size = sizeof(float) * h.cols() * h.rows();

	if (clctx == nullptr) {

		printf( "cl_alloc_from_matrix: clctx == null!\n");
		return 1;
	}

	if (alloc_size > 0) {

		buffer = clCreateBuffer(clctx->ctx(), CL_MEM_READ_WRITE, alloc_size, NULL, &clctx->err);

		if (clctx->err != CL_SUCCESS) {

			printf( "clCreateBuffer failed with %d - %s\n", clctx->err, oclErrorString(clctx->err));
			return 1;

		}

	} else {

		fprintf(stderr, "alloc_matrix: alloc_size <= 0!\n");
	}

	cl_mem_allocated += alloc_size;
	return 0;

}

void cl_free_matrix(cl_matrix& m) {

	clReleaseMemObject((cl_mem)m.device_data);
	cl_mem_allocated -= m.cols() * m.rows() * sizeof(float);

}

int cl_copy_matrix_to_device(cl_ctx* ctx, cl_mem device_data, Eigen::MatrixXf& src) {

	if (ctx == nullptr) {

		printf( "cl_alloc_from_matrix: clctx == null!\n");
		return 1;
	}

	size_t bytes = src.rows() * src.cols() * sizeof(float);

	ctx->err = clEnqueueWriteBuffer(ctx->queue(), device_data, CL_TRUE, 0, bytes, src.data(), 0, NULL, &ctx->event);

	if (ctx->err != CL_SUCCESS) {

		printf( "clEnqueueWriteBuffer failed with %d - %s\n", ctx->err, oclErrorString (ctx->err));

	}

	clWaitForEvents(1, &ctx->event);
	return 0;
}

int cl_copy_matrix_to_host(cl_ctx* ctx, Eigen::MatrixXf& dst, cl_mem device_data) {

	if (ctx == nullptr) {

		printf( "cl_alloc_from_matrix: clctx == null!\n");
		return 1;
	}

	size_t bytes = dst.rows() * dst.cols() * sizeof(float);

	ctx->err = clEnqueueReadBuffer(ctx->queue(), device_data, CL_TRUE, 0, bytes, dst.data(), 0, NULL, NULL);

	if (ctx->err != CL_SUCCESS) {

		printf( "clEnqueueReadBuffer failed with %d - %s\n", ctx->err, oclErrorString (ctx->err) );

	}

	return 0;
}

#endif
