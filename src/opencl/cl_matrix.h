/*
    @Author: kmrocki@us.ibm.com
    @Date:   2017-04-24 16:58:15
    @Last Modified by:   kmrocki@us.ibm.com
    @Last Modified time: 2017-04-29 11:35:06
*/

#include <Eigen/Dense>

#ifndef __CL_MATRIX__
#define __CL_MATRIX__

template<class T>
using host_matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

int cl_copy_device_to_device (cl_ctx* clctx, cl_mem& src, cl_mem& dst, size_t offset_src, size_t offset_dst, size_t bytes, bool wait = false);

template <typename T>
int cl_alloc_from_matrix (cl_ctx* ctx, cl_mem& buffer, host_matrix<T>& h, cl_mem_flags flags = CL_MEM_READ_WRITE);

template <typename T>
int cl_copy_matrix_to_host (cl_ctx* ctx, host_matrix<T>& dst, cl_mem device_data);

template <typename T>
int cl_copy_matrix_to_device (cl_ctx* ctx, cl_mem device_data, host_matrix<T>& src, bool wait = false);

unsigned long cl_mem_allocated = 0L;

template <typename T = float>
class cl_matrix {

  public:

	cl_mem device_data;
	cl_mem& ref_device_data;

	//temp buffers
	cl_mem scratchBuf;

	cl_mem iMax; // index of max value (updated through cl_max_coeff(cl_matrix& m)) - device mem
	cl_uint indexMax = 0; // index of max value (updated through cl_max_coeff(cl_matrix& m))

	cl_mem d_sum; // sum of elements
	T h_sum; // host copy of the sum

	unsigned int lenScratchBuf = 0;

	host_matrix<T> host_data;
	host_matrix<T>& ref_host_data;

	cl_ctx* matrix_ctx;

	cl_matrix (cl_ctx* ctx = nullptr) : ref_device_data (device_data), ref_host_data (host_data), matrix_ctx (ctx) {
		device_data = nullptr;
		scratchBuf = nullptr;
		iMax = nullptr;
	}

	cl_matrix (cl_ctx* ctx, Eigen::Vector2i size) : cl_matrix (ctx) {
		host_data = host_matrix<T> (size[0], size[1]);
		host_data.setZero();
		ref_host_data = host_data;
		alloc_device_mem();

	};

	cl_matrix (cl_ctx* ctx, host_matrix<T>& m) : cl_matrix (ctx) {
		host_data = m;
		ref_host_data = host_data;
		alloc_device_mem();
	};

	cl_matrix& operator= (const cl_matrix& other) {
		host_data = other.host_data;
		ref_host_data = host_data;
		matrix_ctx = other.matrix_ctx;
		free_device_mem();
		alloc_device_mem();
		return *this;
	};

	void resize (size_t rows, size_t cols) {
		host_data.resize (rows, cols);
		free_device_mem();
		alloc_device_mem();
	}

	T sum() { cl_sum(*this, false, true); return h_sum; }

	size_t rows() const {
		return ref_host_data.rows();
	}
	size_t cols() const {
		return ref_host_data.cols();
	}
	size_t length() const {
		return rows() * cols();
	}

	Eigen::Vector2i size() const {
		return { rows(), cols() };
	}

	cl_matrix (const cl_matrix& other) : cl_matrix (other.matrix_ctx, other.size() ) {}

	int setZero (bool wait = false) {
		T zero = T (0);
		std::string func_string = "cl_matrix_set_zero";

		if (matrix_ctx->profiling_enabled) clFinish (matrix_ctx->queue() );

		CL_SAFE_CALL (clEnqueueFillBuffer (matrix_ctx->queue(), device_data, &zero, sizeof (T), 0, length() * sizeof (T), 0, nullptr, &matrix_ctx->cl_events[func_string]) );

		if ( (!matrix_ctx->asynchronous || wait) || matrix_ctx->profiling_enabled) matrix_ctx->get_profiling_data (func_string);

		matrix_ctx->pdata[func_string].bytes_out += rows() * cols() * sizeof (float);
		return 0;
	}

	int setOnes (bool wait = false) {
		T one = T (1);
		std::string func_string = "cl_matrix_set_ones";

		if (matrix_ctx->profiling_enabled) clFinish (matrix_ctx->queue() );

		CL_SAFE_CALL (clEnqueueFillBuffer (matrix_ctx->queue(), device_data, &one, sizeof (T), 0, length() * sizeof (T), 0, nullptr, &matrix_ctx->cl_events[func_string]) );

		if ( (!matrix_ctx->asynchronous || wait) || matrix_ctx->profiling_enabled) matrix_ctx->get_profiling_data (func_string);

		matrix_ctx->pdata[func_string].bytes_out += rows() * cols() * sizeof (float);
		return 0;
	}

	~cl_matrix() {
		free_device_mem();
	};

	void alloc_device_mem() {
		cl_alloc_from_matrix (matrix_ctx, device_data, ref_host_data, CL_MEM_READ_WRITE);
		ref_device_data = device_data;
	}

	void free_device_mem() {
		if (device_data) clReleaseMemObject ( (cl_mem) device_data);
		if (scratchBuf) clReleaseMemObject ( (cl_mem) scratchBuf);
		if (iMax) clReleaseMemObject ( (cl_mem) iMax);
	}

	void sync_device() {
		cl_copy_matrix_to_device (matrix_ctx, device_data, ref_host_data);
	}

	void sync_host() {
		cl_copy_matrix_to_host (matrix_ctx, ref_host_data, device_data);
	}

};

int cl_copy_device_to_device (cl_ctx* ctx, cl_mem& src, cl_mem& dst, size_t offset_src, size_t offset_dst, size_t bytes, bool wait) {

	if (ctx == nullptr) {
		printf ("cl_copy_device_to_device: ctx== null!\n");
		return 1;
	}

	std::string event_string = "cl_copy_device_to_device";

	if (ctx->profiling_enabled) clFinish (ctx->queue() );

	CL_SAFE_CALL (clEnqueueCopyBuffer (ctx->queue(), src, dst, offset_src, offset_dst, bytes, 0, NULL, &ctx->cl_events[event_string]) );

	if ( (!ctx->asynchronous || wait) || ctx->profiling_enabled) ctx->get_profiling_data (event_string);

	return 0;
}

template <typename T = float>
int cl_alloc_from_matrix (cl_ctx* clctx, cl_mem& buffer, host_matrix<T>& h, cl_mem_flags flags) {
	size_t alloc_size = sizeof (T) * h.cols() * h.rows();

	if (clctx == nullptr) {
		printf ("cl_alloc_from_matrix: clctx == null!\n");
		return 1;
	}

	if (alloc_size > 0) {
		buffer = clCreateBuffer (clctx->ctx(), flags, alloc_size, NULL, &clctx->err);

		if (clctx->err != CL_SUCCESS) {
			printf ("clCreateBuffer failed with %d - %s\n", clctx->err, oclErrorString (clctx->err) );
			return 1;
		}

	} else
		fprintf (stderr, "alloc_matrix: alloc_size <= 0!\n");

	cl_mem_allocated += alloc_size;
	return 0;
}

template <typename T = float>
void cl_free_matrix (cl_matrix<T>& m) {
	clReleaseMemObject ( (cl_mem) m.device_data);
	cl_mem_allocated -= m.cols() * m.rows() * sizeof (T);
}

template <typename T = float>
int cl_copy_matrix_to_device (cl_ctx* ctx, cl_mem device_data, host_matrix<T>& src, bool wait) {
	if (ctx == nullptr) {
		printf ("cl_alloc_from_matrix: clctx == null!\n");
		return 1;
	}

	size_t bytes = src.rows() * src.cols() * sizeof (T);
	std::string event_string = "cl_copy_matrix_to_device";

	if (ctx->profiling_enabled) clFinish (ctx->queue() );

	CL_SAFE_CALL (clEnqueueWriteBuffer (ctx->queue(), device_data, CL_TRUE, 0, bytes, src.data(), 0, NULL, &ctx->cl_events[event_string]) );

	if ( (!ctx->asynchronous || wait) || ctx->profiling_enabled) ctx->get_profiling_data (event_string);

	return 0;
}

template <typename T = float>
int cl_copy_matrix_to_host (cl_ctx* ctx, host_matrix<T>& dst, cl_mem device_data) {
	if (ctx == nullptr) {
		printf ("cl_alloc_from_matrix: clctx == null!\n");
		return 1;
	}

	size_t bytes = dst.rows() * dst.cols() * sizeof (T);
	CL_SAFE_CALL (clEnqueueReadBuffer (ctx->queue(), device_data, CL_TRUE, 0, bytes, dst.data(), 0, NULL, NULL) );

	if (ctx->profiling_enabled) clFinish (ctx->queue() );

	return 0;
}

#endif
