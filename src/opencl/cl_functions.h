/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-04-25 08:06:57
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-26 21:12:16
*/

#ifndef __CL_FUNCTIONS__
#define __CL_FUNCTIONS__

#include <opencl/cl_blas_defs.h>
#include <opencl/cl_matrix.h>
#include <opencl/cl_rand.h>

unsigned long cl_flops_performed = 0L;

// void cl_randn ( cl_matrix<float> &x, float mean = 0.0f, float stddev = 1.0f ) {}

void cl_matrix_mult ( cl_matrix<float> &c, cl_matrix<float> &a, cl_matrix<float> &b, bool wait );

void cl_matrix_rand ( cl_matrix<float> &y, bool wait = false ) {

	/* Setup the kernel */
	y.matrix_ctx->err = clSetKernelArg ( y.matrix_ctx->kernels_rand["uniform01"], 0, sizeof( bufIn ),  &bufIn );

	y.matrix_ctx->err = clSetKernelArg ( y.matrix_ctx->kernels_rand["uniform01"], 1, sizeof ( cl_mem ), ( void * ) &y.device_data );

	if ( y.matrix_ctx->err != CL_SUCCESS ) {
		printf ( "cl_matrix_rand : clSetKernelArg failed with %d\n", y.matrix_ctx->err );

	}

	/* Execute the kernel and read back results */
	y.matrix_ctx->err = clEnqueueNDRangeKernel(y.matrix_ctx->queue(), y.matrix_ctx->kernels_rand["uniform01"], 1, NULL, &numWorkItems, NULL, 0, NULL, &y.matrix_ctx->event);

	if ( y.matrix_ctx->err != CL_SUCCESS ) {
		printf ( "cl_matrix_rand : clEnqueueNDRangeKernel failed with %d\n", y.matrix_ctx->err );

	}

	if ( !y.matrix_ctx->asynchronous || wait ) clWaitForEvents ( 1, &y.matrix_ctx->event );

}

void cl_gather_data ( cl_matrix<float> &src, cl_matrix<float> &dst, const cl_matrix<int> &idxs, bool wait = false) {

	unsigned int count = dst.rows() * dst.cols();
	unsigned int n = dst.rows();

	clSetKernelArg ( dst.matrix_ctx->kernels4["gather_data"], 0, sizeof ( cl_mem ), ( void * ) &src.device_data );
	clSetKernelArg ( dst.matrix_ctx->kernels4["gather_data"], 1, sizeof ( cl_mem ), ( void * ) &dst.device_data );
	clSetKernelArg ( dst.matrix_ctx->kernels4["gather_data"], 2, sizeof ( cl_mem ), ( void * ) &idxs.device_data );
	clSetKernelArg ( dst.matrix_ctx->kernels4["gather_data"], 3, sizeof ( unsigned int ), ( void * ) &n );
	clSetKernelArg ( dst.matrix_ctx->kernels4["gather_data"], 4, sizeof ( unsigned int ), ( void * ) &count );

	size_t global_work_size = ( ( count / dst.matrix_ctx->local_work_size ) + 1 ) * dst.matrix_ctx->local_work_size;

	dst.matrix_ctx->err = clEnqueueNDRangeKernel ( dst.matrix_ctx->queue(), dst.matrix_ctx->kernels4["gather_data"], 1, NULL, &global_work_size, &dst.matrix_ctx->local_work_size, 0, NULL, &dst.matrix_ctx->event );

	if ( !dst.matrix_ctx->asynchronous || wait ) clWaitForEvents ( 1, &dst.matrix_ctx->event );

}

// y = f(y)
void cl_elementwise ( cl_matrix<float> &y, std::string func, bool wait = false ) {

	unsigned int count = y.rows() * y.cols();

	clSetKernelArg ( y.matrix_ctx->kernels1[func], 0, sizeof ( cl_mem ), ( void * ) &y.device_data );
	clSetKernelArg ( y.matrix_ctx->kernels1[func], 1, sizeof ( unsigned int ), ( void * ) &count );

	size_t global_work_size = ( ( count / y.matrix_ctx->local_work_size ) + 1 ) * y.matrix_ctx->local_work_size;

	y.matrix_ctx->err = clEnqueueNDRangeKernel ( y.matrix_ctx->queue(), y.matrix_ctx->kernels1[func], 1, NULL, &global_work_size, &y.matrix_ctx->local_work_size, 0, NULL, &y.matrix_ctx->event );

	if ( !y.matrix_ctx->asynchronous || wait ) clWaitForEvents ( 1, &y.matrix_ctx->event );

}

// y = f(x)
void cl_elementwise ( cl_matrix<float> &y, cl_matrix<float> &x, std::string func, bool wait = false ) {

	unsigned int count = y.rows() * y.cols();

	clSetKernelArg ( y.matrix_ctx->kernels2[func], 0, sizeof ( cl_mem ), ( void * ) &y.device_data );
	clSetKernelArg ( y.matrix_ctx->kernels2[func], 1, sizeof ( cl_mem ), ( void * ) &x.device_data );
	clSetKernelArg ( y.matrix_ctx->kernels2[func], 2, sizeof ( unsigned int ), ( void * ) &count );

	size_t global_work_size = ( ( count / y.matrix_ctx->local_work_size ) + 1 ) * y.matrix_ctx->local_work_size;

	y.matrix_ctx->err = clEnqueueNDRangeKernel ( y.matrix_ctx->queue(), y.matrix_ctx->kernels2[func], 1, NULL, &global_work_size, &y.matrix_ctx->local_work_size, 0, NULL, &y.matrix_ctx->event );

	if ( !y.matrix_ctx->asynchronous || wait ) clWaitForEvents ( 1, &y.matrix_ctx->event );

}

// y = x op z
void cl_elementwise ( cl_matrix<float> &y, cl_matrix<float> &x, cl_matrix<float> &z, std::string func, bool wait = false ) {

	unsigned int count = y.rows() * y.cols();

	clSetKernelArg ( y.matrix_ctx->kernels3[func], 0, sizeof ( cl_mem ), ( void * ) &y.device_data );
	clSetKernelArg ( y.matrix_ctx->kernels3[func], 1, sizeof ( cl_mem ), ( void * ) &x.device_data );
	clSetKernelArg ( y.matrix_ctx->kernels3[func], 2, sizeof ( cl_mem ), ( void * ) &z.device_data );
	clSetKernelArg ( y.matrix_ctx->kernels3[func], 3, sizeof ( unsigned int ), ( void * ) &count );

	size_t global_work_size = ( ( count / y.matrix_ctx->local_work_size ) + 1 ) * y.matrix_ctx->local_work_size;

	y.matrix_ctx->err = clEnqueueNDRangeKernel ( y.matrix_ctx->queue(), y.matrix_ctx->kernels3[func], 1, NULL, &global_work_size, &y.matrix_ctx->local_work_size, 0, NULL, &y.matrix_ctx->event );

	if ( !y.matrix_ctx->asynchronous || wait ) clWaitForEvents ( 1, &y.matrix_ctx->event );

}

// y = x op z
void cl_elementwise ( cl_matrix<float> &y, cl_matrix<float> &x, float z, std::string func, bool wait = false ) {

	unsigned int count = y.rows() * y.cols();
	float local_z = z;

	clSetKernelArg ( y.matrix_ctx->kernels3[func], 0, sizeof ( cl_mem ), ( void * ) &y.device_data );
	clSetKernelArg ( y.matrix_ctx->kernels3[func], 1, sizeof ( cl_mem ), ( void * ) &x.device_data );
	clSetKernelArg ( y.matrix_ctx->kernels3[func], 2, sizeof ( cl_float ), ( void * ) &local_z );
	clSetKernelArg ( y.matrix_ctx->kernels3[func], 3, sizeof ( unsigned int ), ( void * ) &count );

	size_t global_work_size = ( ( count / y.matrix_ctx->local_work_size ) + 1 ) * y.matrix_ctx->local_work_size;

	y.matrix_ctx->err = clEnqueueNDRangeKernel ( y.matrix_ctx->queue(), y.matrix_ctx->kernels3[func], 1, NULL, &global_work_size, &y.matrix_ctx->local_work_size, 0, NULL, &y.matrix_ctx->event );

	if ( !y.matrix_ctx->asynchronous || wait ) clWaitForEvents ( 1, &y.matrix_ctx->event );

}

void cl_matrix_scalar ( cl_matrix<float> &y, std::string func, bool wait = false ) {

	unsigned int count = y.rows() * y.cols();

	clSetKernelArg ( y.matrix_ctx->kernels_mat_scalar[func], 0, sizeof ( cl_mem ), ( void * ) &y.device_data );
	clSetKernelArg ( y.matrix_ctx->kernels_mat_scalar[func], 1, sizeof ( cl_mem ), ( void * ) &y.device_data );
	clSetKernelArg ( y.matrix_ctx->kernels_mat_scalar[func], 2, sizeof ( unsigned int ), ( void * ) &y.indexMax );
	clSetKernelArg ( y.matrix_ctx->kernels_mat_scalar[func], 3, sizeof ( unsigned int ), ( void * ) &count );

	size_t global_work_size = ( ( count / y.matrix_ctx->local_work_size ) + 1 ) * y.matrix_ctx->local_work_size;

	y.matrix_ctx->err = clEnqueueNDRangeKernel ( y.matrix_ctx->queue(), y.matrix_ctx->kernels_mat_scalar[func], 1, NULL, &global_work_size, &y.matrix_ctx->local_work_size, 0, NULL, &y.matrix_ctx->event );

	if ( !y.matrix_ctx->asynchronous || wait ) clWaitForEvents ( 1, &y.matrix_ctx->event );

}

int cl_max_coeff ( cl_matrix<float> &m, bool wait = false, bool read_to_hostmem = false ) {

	size_t N = m.rows() * m.cols();

	if ( m.lenScratchBuf < N ) {

		m.lenScratchBuf = N;
		if ( m.scratchBuf ) clReleaseMemObject ( ( cl_mem ) m.scratchBuf );
		m.scratchBuf = clCreateBuffer ( m.matrix_ctx->ctx(), CL_MEM_READ_WRITE, ( N * sizeof ( cl_float ) * 2 ), NULL, &m.matrix_ctx->err );

		if ( m.matrix_ctx->err != CL_SUCCESS ) {
			printf ( "cl_max_coeff : m.scratchBuf = clCreateBuffer() failed with %d\n", m.matrix_ctx->err );
			return 1;

		}

	}

	if ( !m.iMax ) m.iMax = clCreateBuffer ( m.matrix_ctx->ctx(), CL_MEM_READ_WRITE, sizeof ( cl_uint ), NULL, &m.matrix_ctx->err );

	if ( m.matrix_ctx->err != CL_SUCCESS ) {
		printf ( "cl_max_coeff : m.iMax = clCreateBuffer() failed with %d\n", m.matrix_ctx->err );
		return 1;

	}


	// CL_BLAS_STATUS_TYPE status = CL_BLAS_ISAMAX ( N, m.iMax, 0, m.device_data, 0, 1, m.scratchBuf, 1, &m.matrix_ctx->queue(), 0, NULL, &m.matrix_ctx->event );

	CL_BLAS_STATUS_TYPE status = CL_BLAS_ISAMAX (  N, m.iMax, 0, m.device_data, 0, 1, m.scratchBuf, 1, &m.matrix_ctx->queue(), 0, NULL, &m.matrix_ctx->event );

	if ( status != CL_BLAS_SUCCESS_CODE ) {
		printf ( "clblasiSamax() failed with %d\n", m.matrix_ctx->err );
		return 1;

	} else {

		/* Wait for calculations to be finished. */
		if ( !m.matrix_ctx->asynchronous || wait ) clWaitForEvents ( 1, &m.matrix_ctx->event );

		/* Fetch results of calculations from GPU memory. */
		if ( read_to_hostmem ) {

			m.matrix_ctx->err = clEnqueueReadBuffer ( m.matrix_ctx->queue(), m.iMax, CL_TRUE, 0, sizeof ( cl_uint ), &m.indexMax, 0, NULL, NULL );

			if ( m.matrix_ctx->err != CL_SUCCESS ) {
				printf ( "cl_max_coeff : clEnqueueReadBuffer(m.matrix_ctx->queue(), m.iMax failed with %d\n", m.matrix_ctx->err );
				return 1;

			}

			m.indexMax -= 1;

		}

	}

	return 0;
}

void cl_sub_max_coeff ( cl_matrix<float> &m, bool wait = false ) {

	cl_max_coeff ( m, wait );
	cl_matrix_scalar ( m, "sub", wait );

}

void cl_colsumdiv ( cl_matrix<float> &y, cl_matrix<float> &x, bool wait = false ) {

	unsigned int count = x.rows() * x.cols();
	unsigned int cols = y.cols();

	clSetKernelArg ( y.matrix_ctx->kernels2["colsumdiv"], 0, sizeof ( cl_mem ), ( void * ) &y.device_data );
	clSetKernelArg ( y.matrix_ctx->kernels2["colsumdiv"], 1, sizeof ( cl_mem ), ( void * ) &x.device_data );

	clSetKernelArg ( y.matrix_ctx->kernels2["colsumdiv"], 2, sizeof ( unsigned int ), ( void * ) &count );
	clSetKernelArg ( y.matrix_ctx->kernels2["colsumdiv"], 3, sizeof ( unsigned int ), ( void * ) &cols );

	size_t global_work_size = ( ( cols / y.matrix_ctx->local_work_size ) + 1 ) * y.matrix_ctx->local_work_size;

	y.matrix_ctx->err = clEnqueueNDRangeKernel ( y.matrix_ctx->queue(), y.matrix_ctx->kernels2["colsumdiv"], 1, NULL, &global_work_size, &y.matrix_ctx->local_work_size, 0, NULL, &y.matrix_ctx->event );

	if ( !y.matrix_ctx->asynchronous || wait ) clWaitForEvents ( 1, &y.matrix_ctx->event );

}

cl_matrix<float> colsums;
cl_matrix<float> ones_column;

void cl_softmax ( cl_matrix<float> &y, cl_matrix<float> &x, bool wait = false ) {

	cl_elementwise ( y, x, "exp", wait );

	if ( !colsums.device_data ) {

		colsums = cl_matrix<float> ( y.matrix_ctx, {1, y.cols() } );

	}

	//TODO: split and improve
	cl_colsumdiv ( colsums, y, wait );

}

void cl_matrix_mult ( cl_matrix<float> &c, cl_matrix<float> &a, cl_matrix<float> &b, bool tA, bool tB, float alpha, float beta, bool wait = false ) {

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

	status = CL_BLAS_SGEMM ( order, transA, transB, M, N, K, alpha, ( cl_mem ) a.device_data, offset_a, lda, ( cl_mem ) b.device_data, offset_b, ldb, beta, ( cl_mem ) c.device_data, offset_c, ldc, 1, &c.matrix_ctx->queue(), 0, NULL, &c.matrix_ctx->event );

	if ( status != CL_BLAS_SUCCESS_CODE )
		printf ( "clblasSgemm() failed with %d - %s\n", c.matrix_ctx->err, oclErrorString ( c.matrix_ctx->err ) );

	else {

		if ( !c.matrix_ctx->asynchronous || wait ) clWaitForEvents ( 1, &c.matrix_ctx->event );

	}

	cl_flops_performed += M * N * K * 2;

}

#endif