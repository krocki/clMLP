/*
    @Author: kmrocki@us.ibm.com
    @Date:   2017-04-25 08:06:57
    @Last Modified by:   kmrocki@us.ibm.com
    @Last Modified time: 2017-04-29 11:33:28
*/

#ifndef __CL_FUNCTIONS__
#define __CL_FUNCTIONS__

#include <opencl/cl_blas_defs.h>
#include <opencl/cl_matrix.h>
#include <opencl/cl_rand.h>

int cl_matrix_mult (cl_matrix<float>& c, cl_matrix<float>& a, cl_matrix<float>& b, bool wait);

void cl_matrix_randn (cl_matrix<float>& y, bool wait = false) {

	unsigned int count = y.rows() * y.cols();
	unsigned int n = count / numWorkItems + 1;

	/* Setup the kernel */
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["normal"], 0, sizeof (bufInNormal),  &bufInNormal) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["normal"], 1, sizeof (normalDist_buffer),  &normalDist_buffer) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["normal"], 2, sizeof (cl_mem), (void*) &y.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["normal"], 3, sizeof (unsigned int), (void*) &n) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["normal"], 4, sizeof (unsigned int), (void*) &count) );

	/* Execute the kernel and read back results */
	std::string func_string = "cl_matrix_randn_" + std::to_string(y.rows()) + "_" + std::to_string(y.cols());

	if (y.matrix_ctx->profiling_enabled) clFinish (y.matrix_ctx->queue() );

	CL_SAFE_CALL (clEnqueueNDRangeKernel (y.matrix_ctx->queue(), y.matrix_ctx->kernels_rand["normal"], 1, NULL, &numWorkItems, NULL, 0, NULL, &y.matrix_ctx->cl_events[func_string]) );

	if ( (!y.matrix_ctx->asynchronous || wait) || y.matrix_ctx->profiling_enabled) y.matrix_ctx->get_profiling_data (func_string);
}

void cl_matrix_rand (cl_matrix<float>& y, bool wait = false) {

	unsigned int count = y.rows() * y.cols();
	unsigned int n = count / numWorkItems + 1;

	/* Setup the kernel */
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["uniform01"], 0, sizeof (bufInUniform),  &bufInUniform) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["uniform01"], 1, sizeof (cl_mem), (void*) &y.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["uniform01"], 2, sizeof (unsigned int), (void*) &n) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["uniform01"], 3, sizeof (unsigned int), (void*) &count) );
	std::string func_string = "cl_matrix_rand_" + std::to_string(count) + "_" + std::to_string(y.rows()) + "x" + std::to_string(y.cols());

	if (y.matrix_ctx->profiling_enabled) clFinish (y.matrix_ctx->queue() );

	/* Execute the kernel and read back results */
	CL_SAFE_CALL (clEnqueueNDRangeKernel (y.matrix_ctx->queue(), y.matrix_ctx->kernels_rand["uniform01"], 1, NULL, &numWorkItems, NULL, 0, NULL, &y.matrix_ctx->cl_events[func_string]) );

	if ( (!y.matrix_ctx->asynchronous || wait) || y.matrix_ctx->profiling_enabled) y.matrix_ctx->get_profiling_data (func_string);
}

void cl_matrix_randi (cl_matrix<int>& y, int range_min = 0, int range_max = 99, bool wait = false) {

	unsigned int count = y.rows() * y.cols();
	unsigned int n = count / numWorkItems + 1;
	int r_min = range_min;
	int r_max = range_max;

	if (r_max <= r_min)  printf ("r_max <= r_min : r_min = %d r_max = %d!\n", r_min, r_max);

	/* Setup the kernel */
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["randi"], 0, sizeof (bufInUniform),  &bufInUniform) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["randi"], 1, sizeof (cl_mem), (void*) &y.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["randi"], 2, sizeof (int), (void*) &r_min) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["randi"], 3, sizeof (int), (void*) &r_max) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["randi"], 4, sizeof (unsigned int), (void*) &n) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_rand["randi"], 5, sizeof (unsigned int), (void*) &count) );

	/* Execute the kernel */
	std::string func_string = "cl_matrix_randi_" + std::to_string(count) + "_" + std::to_string(y.rows()) + "x" + std::to_string(y.cols());

	if (y.matrix_ctx->profiling_enabled) clFinish (y.matrix_ctx->queue() );

	CL_SAFE_CALL (clEnqueueNDRangeKernel (y.matrix_ctx->queue(), y.matrix_ctx->kernels_rand["randi"], 1, NULL, &numWorkItems, NULL, 0, NULL, &y.matrix_ctx->cl_events[func_string]) );

	if ( (!y.matrix_ctx->asynchronous || wait) || y.matrix_ctx->profiling_enabled) y.matrix_ctx->get_profiling_data (func_string);
}

void cl_gather_data (cl_matrix<float>& src, cl_matrix<float>& dst, const cl_matrix<int>& idxs, bool wait = false) {

	unsigned int count = dst.rows() * dst.cols();
	unsigned int n = dst.rows();

	CL_SAFE_CALL (clSetKernelArg (dst.matrix_ctx->kernels4["gather_data"], 0, sizeof (cl_mem), (void*) &src.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (dst.matrix_ctx->kernels4["gather_data"], 1, sizeof (cl_mem), (void*) &dst.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (dst.matrix_ctx->kernels4["gather_data"], 2, sizeof (cl_mem), (void*) &idxs.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (dst.matrix_ctx->kernels4["gather_data"], 3, sizeof (unsigned int), (void*) &n) );
	CL_SAFE_CALL (clSetKernelArg (dst.matrix_ctx->kernels4["gather_data"], 4, sizeof (unsigned int), (void*) &count) );

	size_t global_work_size = ( (count / dst.matrix_ctx->local_work_size) + 1) * dst.matrix_ctx->local_work_size;
	std::string func_string = "cl_gather_data_" + std::to_string(count) + "_" + std::to_string(dst.rows()) + "x" + std::to_string(dst.cols());

	if (dst.matrix_ctx->profiling_enabled) clFinish (dst.matrix_ctx->queue() );

	/* Execute the kernel */
	CL_SAFE_CALL (clEnqueueNDRangeKernel (dst.matrix_ctx->queue(), dst.matrix_ctx->kernels4["gather_data"], 1, NULL, &global_work_size, &dst.matrix_ctx->local_work_size, 0, NULL, &dst.matrix_ctx->cl_events[func_string]) );

	if ( (!dst.matrix_ctx->asynchronous || wait) || dst.matrix_ctx->profiling_enabled) dst.matrix_ctx->get_profiling_data (func_string);
}

// y = f(y)
void cl_elementwise (cl_matrix<float>& y, std::string func, bool wait = false) {

	unsigned int count = y.rows() * y.cols();

	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels1[func], 0, sizeof (cl_mem), (void*) &y.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels1[func], 1, sizeof (unsigned int), (void*) &count) );

	size_t global_work_size = ( (count / y.matrix_ctx->local_work_size) + 1) * y.matrix_ctx->local_work_size;
	std::string func_string = "cl_elementwise_1_" + func + "_" + std::to_string(count) + "_" + std::to_string(y.rows()) + "x" + std::to_string(y.cols());

	if (y.matrix_ctx->profiling_enabled) clFinish (y.matrix_ctx->queue() );

	/* Execute the kernel */
	CL_SAFE_CALL (clEnqueueNDRangeKernel (y.matrix_ctx->queue(), y.matrix_ctx->kernels1[func], 1, NULL, &global_work_size, &y.matrix_ctx->local_work_size, 0, NULL, &y.matrix_ctx->cl_events[func_string]) );

	if ( (!y.matrix_ctx->asynchronous || wait) || y.matrix_ctx->profiling_enabled) y.matrix_ctx->get_profiling_data (func_string);

	y.matrix_ctx->pdata[func_string].flops += y.rows() * y.cols();
	y.matrix_ctx->pdata[func_string].bytes_out += y.rows() * y.cols() * sizeof (float);
	y.matrix_ctx->pdata[func_string].bytes_in += y.rows() * y.cols() * sizeof (float);
}

// y = f(x)
void cl_elementwise (cl_matrix<float>& y, cl_matrix<float>& x, std::string func, bool wait = false) {

	unsigned int count = y.rows() * y.cols();

	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels2[func], 0, sizeof (cl_mem), (void*) &y.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels2[func], 1, sizeof (cl_mem), (void*) &x.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels2[func], 2, sizeof (unsigned int), (void*) &count) );

	size_t global_work_size = ( (count / y.matrix_ctx->local_work_size) + 1) * y.matrix_ctx->local_work_size;
	std::string func_string = "cl_elementwise_2_" + func + "_" + std::to_string(count) + "_" + std::to_string(y.rows()) + "x" + std::to_string(y.cols());

	if (y.matrix_ctx->profiling_enabled) clFinish (y.matrix_ctx->queue() );

	/* Execute the kernel */
	CL_SAFE_CALL (clEnqueueNDRangeKernel (y.matrix_ctx->queue(), y.matrix_ctx->kernels2[func], 1, NULL, &global_work_size, &y.matrix_ctx->local_work_size, 0, NULL, &y.matrix_ctx->cl_events[func_string]) );

	if ( (!y.matrix_ctx->asynchronous || wait) || y.matrix_ctx->profiling_enabled) y.matrix_ctx->get_profiling_data (func_string);

	y.matrix_ctx->pdata[func_string].flops += y.rows() * y.cols();
	y.matrix_ctx->pdata[func_string].bytes_out += y.rows() * y.cols() * sizeof (float);
	y.matrix_ctx->pdata[func_string].bytes_in += x.rows() * x.cols() * sizeof (float);
}

// y = x op z
void cl_elementwise (cl_matrix<float>& y, cl_matrix<float>& x, cl_matrix<float>& z, std::string func, bool wait = false) {

	unsigned int count = y.rows() * y.cols();
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels3[func], 0, sizeof (cl_mem), (void*) &y.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels3[func], 1, sizeof (cl_mem), (void*) &x.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels3[func], 2, sizeof (cl_mem), (void*) &z.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels3[func], 3, sizeof (unsigned int), (void*) &count) );

	size_t global_work_size = ( (count / y.matrix_ctx->local_work_size) + 1) * y.matrix_ctx->local_work_size;
	std::string func_string = "cl_elementwise_3_" + func + "_" + std::to_string(count) + "_" + std::to_string(y.rows()) + "x" + std::to_string(y.cols());

	if (y.matrix_ctx->profiling_enabled) clFinish (y.matrix_ctx->queue() );

	/* Execute the kernel */
	CL_SAFE_CALL (clEnqueueNDRangeKernel (y.matrix_ctx->queue(), y.matrix_ctx->kernels3[func], 1, NULL, &global_work_size, &y.matrix_ctx->local_work_size, 0, NULL, &y.matrix_ctx->cl_events[func_string]) );

	if ( (!y.matrix_ctx->asynchronous || wait) || y.matrix_ctx->profiling_enabled) y.matrix_ctx->get_profiling_data (func_string);
}

// y = x op z
void cl_elementwise (cl_matrix<float>& y, cl_matrix<float>& x, float z, std::string func, bool wait = false) {

	unsigned int count = y.rows() * y.cols();
	float local_z = z;

	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels3[func], 0, sizeof (cl_mem), (void*) &y.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels3[func], 1, sizeof (cl_mem), (void*) &x.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels3[func], 2, sizeof (cl_float), (void*) &local_z) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels3[func], 3, sizeof (unsigned int), (void*) &count) );
	size_t global_work_size = ( (count / y.matrix_ctx->local_work_size) + 1) * y.matrix_ctx->local_work_size;
	std::string func_string = "cl_elementwise_3s_" + func + "_" + std::to_string(count) + "_" + std::to_string(y.rows()) + "x" + std::to_string(y.cols());

	if (y.matrix_ctx->profiling_enabled) clFinish (y.matrix_ctx->queue() );

	cl_kernel& kernel = y.matrix_ctx->kernels3[func];

	/* Execute the kernel */
	CL_SAFE_CALL (clEnqueueNDRangeKernel (y.matrix_ctx->queue(), kernel, 1, NULL, &global_work_size, &y.matrix_ctx->local_work_size, 0, NULL, &y.matrix_ctx->cl_events[func_string]) );

	if ( (!y.matrix_ctx->asynchronous || wait) || y.matrix_ctx->profiling_enabled) y.matrix_ctx->get_profiling_data (func_string);

	y.matrix_ctx->pdata[func_string].flops += y.rows() * y.cols();
	y.matrix_ctx->pdata[func_string].bytes_out += y.rows() * y.cols() * sizeof (float);
	y.matrix_ctx->pdata[func_string].bytes_in += x.rows() * x.cols() * sizeof (float);
}

void cl_matrix_scalar (cl_matrix<float>& y, std::string func, bool wait = false) {

	unsigned int count = y.rows() * y.cols();

	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_mat_scalar[func], 0, sizeof (cl_mem), (void*) &y.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_mat_scalar[func], 1, sizeof (cl_mem), (void*) &y.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_mat_scalar[func], 2, sizeof (unsigned int), (void*) &y.indexMax) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels_mat_scalar[func], 3, sizeof (unsigned int), (void*) &count) );

	size_t global_work_size = ( (count / y.matrix_ctx->local_work_size) + 1) * y.matrix_ctx->local_work_size;
	std::string func_string = "cl_elementwise_1s_" + func + "_" + std::to_string(count) + "_" + std::to_string(y.rows()) + "x" + std::to_string(y.cols());

	if (y.matrix_ctx->profiling_enabled) clFinish (y.matrix_ctx->queue() );

	/* Execute the kernel */
	CL_SAFE_CALL (clEnqueueNDRangeKernel (y.matrix_ctx->queue(), y.matrix_ctx->kernels_mat_scalar[func], 1, NULL, &global_work_size, &y.matrix_ctx->local_work_size, 0, NULL, &y.matrix_ctx->cl_events[func_string]) );

	if ( (!y.matrix_ctx->asynchronous || wait) || y.matrix_ctx->profiling_enabled) y.matrix_ctx->get_profiling_data (func_string);
}

int cl_sum(cl_matrix<float>& m, bool wait = false, bool read_to_hostmem = false) {

	size_t N = m.rows() * m.cols();

	if (m.lenScratchBuf < N) {
		m.lenScratchBuf = N;

		if (m.scratchBuf) clReleaseMemObject ( (cl_mem) m.scratchBuf);

		m.scratchBuf = clCreateBuffer (m.matrix_ctx->ctx(), CL_MEM_READ_WRITE, (N * sizeof (cl_float) * 2), NULL, &m.matrix_ctx->err);

		if (m.matrix_ctx->err != CL_SUCCESS) {
			printf ("cl_max_coeff : m.scratchBuf = clCreateBuffer() failed with %d\n", m.matrix_ctx->err);
			return 1;
		}
	}

	if (!m.d_sum) m.d_sum = clCreateBuffer (m.matrix_ctx->ctx(), CL_MEM_READ_WRITE, sizeof (cl_float), NULL, &m.matrix_ctx->err);

	if (m.matrix_ctx->err != CL_SUCCESS) {
		printf ("cl_sum : m.d_sum = clCreateBuffer() failed with %d\n", m.matrix_ctx->err);
		return 1;
	}

	// _CL_TIMED_CALL_
	std::string func_string = "clblas_sasum_" + std::to_string(N) + "_" + std::to_string(m.rows()) + "x" + std::to_string(m.cols());

	if (m.matrix_ctx->profiling_enabled) clFinish (m.matrix_ctx->queue() );

	CL_BLAS_STATUS_TYPE status = CL_BLAS_SASUM (N, m.d_sum, 0, m.ref_device_data, 0, 1, m.scratchBuf, 1, &m.matrix_ctx->queue(), 0, NULL, &m.matrix_ctx->cl_events[func_string]);

	if (status != CL_BLAS_SUCCESS_CODE) {
		printf ("clblas_sasum() failed with %d - %s\n", status, oclErrorString (status) );
		return 1;

	} else {
		/* Wait for calculations to be finished. */
		if ( (!m.matrix_ctx->asynchronous || wait) || m.matrix_ctx->profiling_enabled) m.matrix_ctx->get_profiling_data (func_string);

		/* Fetch results of calculations from GPU memory. */
		if (read_to_hostmem) {
			CL_SAFE_CALL (clEnqueueReadBuffer (m.matrix_ctx->queue(), m.d_sum, CL_TRUE, 0, sizeof (cl_float), &m.h_sum, 0, NULL, NULL) );
		}
	}

	return 0;
}
int cl_max_coeff (cl_matrix<float>& m, bool wait = false, bool read_to_hostmem = false) {
	size_t N = m.rows() * m.cols();

	if (m.lenScratchBuf < N) {
		m.lenScratchBuf = N;

		if (m.scratchBuf) clReleaseMemObject ( (cl_mem) m.scratchBuf);

		m.scratchBuf = clCreateBuffer (m.matrix_ctx->ctx(), CL_MEM_READ_WRITE, (N * sizeof (cl_float) * 2), NULL, &m.matrix_ctx->err);

		if (m.matrix_ctx->err != CL_SUCCESS) {
			printf ("cl_max_coeff : m.scratchBuf = clCreateBuffer() failed with %d\n", m.matrix_ctx->err);
			return 1;
		}
	}

	if (!m.iMax) m.iMax = clCreateBuffer (m.matrix_ctx->ctx(), CL_MEM_READ_WRITE, sizeof (cl_uint), NULL, &m.matrix_ctx->err);

	if (m.matrix_ctx->err != CL_SUCCESS) {
		printf ("cl_max_coeff : m.iMax = clCreateBuffer() failed with %d\n", m.matrix_ctx->err);
		return 1;
	}

	// _CL_TIMED_CALL_
	std::string func_string = "clblas_isamax_" + std::to_string(N) + "_" + std::to_string(m.rows()) + "x" + std::to_string(m.cols());

	if (m.matrix_ctx->profiling_enabled) clFinish (m.matrix_ctx->queue() );

	CL_BLAS_STATUS_TYPE status = CL_BLAS_ISAMAX (N, m.iMax, 0, m.ref_device_data, 0, 1, m.scratchBuf, 1, &m.matrix_ctx->queue(), 0, NULL, &m.matrix_ctx->cl_events[func_string]);

	if (status != CL_BLAS_SUCCESS_CODE) {
		printf ("clblasiSamax() failed with %d - %s\n", status, oclErrorString (status) );
		return 1;

	} else {
		/* Wait for calculations to be finished. */
		if ( (!m.matrix_ctx->asynchronous || wait) || m.matrix_ctx->profiling_enabled) m.matrix_ctx->get_profiling_data (func_string);

		/* Fetch results of calculations from GPU memory. */
		if (read_to_hostmem) {
			CL_SAFE_CALL (clEnqueueReadBuffer (m.matrix_ctx->queue(), m.iMax, CL_TRUE, 0, sizeof (cl_uint), &m.indexMax, 0, NULL, NULL) );
			m.indexMax -= 1;
		}
	}

	return 0;
}

void cl_sub_max_coeff (cl_matrix<float>& m, bool wait = false) {
	cl_max_coeff (m, wait);
	cl_matrix_scalar (m, "sub", wait);
}

void cl_colsumdiv (cl_matrix<float>& y, cl_matrix<float>& x, bool wait = false) {

	unsigned int count = x.rows() * x.cols();
	unsigned int cols = y.cols();

	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels2["colsumdiv"], 0, sizeof (cl_mem), (void*) &y.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels2["colsumdiv"], 1, sizeof (cl_mem), (void*) &x.ref_device_data) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels2["colsumdiv"], 2, sizeof (unsigned int), (void*) &count) );
	CL_SAFE_CALL (clSetKernelArg (y.matrix_ctx->kernels2["colsumdiv"], 3, sizeof (unsigned int), (void*) &cols) );

	size_t global_work_size = ( (cols / y.matrix_ctx->local_work_size) + 1) * y.matrix_ctx->local_work_size;
	std::string func_string = "cl_colsumdiv_" + std::to_string(count) + "_" + std::to_string(x.rows()) + "x" + std::to_string(x.cols());

	if (y.matrix_ctx->profiling_enabled) clFinish (y.matrix_ctx->queue() );

	/* Execute the kernel */
	CL_SAFE_CALL (clEnqueueNDRangeKernel (y.matrix_ctx->queue(), y.matrix_ctx->kernels2["colsumdiv"], 1, NULL, &global_work_size, &y.matrix_ctx->local_work_size, 0, NULL, &y.matrix_ctx->cl_events[func_string]) );

	if ( (!y.matrix_ctx->asynchronous || wait) || y.matrix_ctx->profiling_enabled) y.matrix_ctx->get_profiling_data (func_string);
}

cl_matrix<float> colsums;
cl_matrix<float> ones_column;

void cl_softmax (cl_matrix<float>& y, cl_matrix<float>& x, bool wait = false) {
	cl_elementwise (y, x, "exp", wait);

	if (!colsums.ref_device_data) {
		colsums = cl_matrix<float> (y.matrix_ctx, {1, y.cols() });
	}

	//TODO: split and improve
	cl_colsumdiv (colsums, y, wait);
}

int cl_matrix_mult (cl_matrix<float>& c, cl_matrix<float>& a, cl_matrix<float>& b, bool tA, bool tB, float alpha, float beta, cl_command_queue aux_queue = nullptr, bool wait = false) {

	const CL_BLAS_MATRIX_ORDER order = CL_BLAS_MATRIX_ORDER_COLUMN;
	const CL_BLAS_MATRIX_TRANSPOSE transA = tA ? CL_BLAS_MATRIX_TRANSPOSED : CL_BLAS_MATRIX_NOT_TRANSPOSED;
	const CL_BLAS_MATRIX_TRANSPOSE transB = tB ? CL_BLAS_MATRIX_TRANSPOSED : CL_BLAS_MATRIX_NOT_TRANSPOSED;
	size_t M = c.rows();
	size_t N = c.cols();
	size_t K = tA ? b.rows() : a.cols();
	size_t lda = tA ? K : M;
	size_t ldb = tB ? N : K;
	size_t ldc = M;
	size_t offset_a = 0;
	size_t offset_b = 0;
	size_t offset_c = 0;
	CL_BLAS_STATUS_TYPE status;
	std::string func_string = "cl_matrix_mult_M_" + std::to_string(M) + "_N_" + std::to_string(N) + "_K_" + std::to_string(K) + "_aT_" + std::to_string(tA) + "_bT_" + std::to_string(tB);

	cl_command_queue exec_queue = aux_queue == nullptr ? c.matrix_ctx->queue() : aux_queue;

	if (c.matrix_ctx->profiling_enabled) clFinish (c.matrix_ctx->queue() );

	/* Execute the kernel */
	status = CL_BLAS_SGEMM (order, transA, transB, M, N, K, alpha, (cl_mem) a.ref_device_data, offset_a, lda, (cl_mem) b.ref_device_data, offset_b, ldb, beta, (cl_mem) c.ref_device_data, offset_c, ldc, 1, &exec_queue, 0, NULL, &c.matrix_ctx->cl_events[func_string]);

	if (status != CL_BLAS_SUCCESS_CODE) {
		printf ("clblasSgemm() failed with %d - %s\n", status, oclErrorString (status) );
		return 1;

	} else {
		if ( (!c.matrix_ctx->asynchronous || wait) || c.matrix_ctx->profiling_enabled) c.matrix_ctx->get_profiling_data (func_string);

		c.matrix_ctx->pdata[func_string].flops += M * N * K * 2;
		c.matrix_ctx->pdata[func_string].bytes_out += c.rows() * c.cols() * sizeof (float);
		c.matrix_ctx->pdata[func_string].bytes_in += (a.rows() * a.cols() + b.rows() * b.cols() ) * sizeof (float);
		return 0;
	}
}

#endif