
// #pragma OPENCL EXTENSION cl_amd_printf : enable

#define FAST_MATH

#ifdef FAST_MATH
#define exp_function native_exp
#define log_function native_log
#define log2_function native_log2
#define sqrt_function native_sqrt
#define rsqrt_function native_rsqrt
#define powr_function native_powr
#define divide_function native_divide
#define fmad_function(a,b,c) (mad ((a), (b), (c)))
#define logistic_function(x) (native_recip ( 1.0f + native_exp ( -(x) ) ))
#else
#define exp_function exp
#define log_function log
#define log2_function log2
#define sqrt_function sqrt
#define rsqrt_function rsqrt
#define powr_function powr
#define divide_function divide
#define fmad_function(a,b,c) ((c) + (a) * (b))
#define logistic_function(x) (1.0f / ( 1.0f + exp ( -(x) ) ))
#endif

#include "isamax.cl"

// http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/opencl-optimization-guide/
// Instruction Throughput (Operations/Cycle for Each Stream Processor)

__kernel void logistic1 (__global float* restrict inputoutput, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) inputoutput[i] = logistic_function (inputoutput[i]);
}

__kernel void logistic2 (__global float* restrict output, __global float* restrict input, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) output[i] = logistic_function (input[i]);
}

__kernel void rectify1 (__global float* restrict inputoutput, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) inputoutput[i] = fmax (inputoutput[i], 0.0f);
}

__kernel void rectify2 (__global float* restrict output, __global float* restrict input, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) output[i] = fmax (input[i], 0.0f);
}

__kernel void expf1 (__global float* restrict inputoutput, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) inputoutput[i] = exp_function (inputoutput[i]);
}

__kernel void expf2 (__global float* restrict output, __global float* restrict input, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) output[i] = exp_function (input[i]);
}

__kernel void sub1 (__global float* restrict output, __global float* restrict input, const unsigned int val, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) output[i] -= input[val];
}

__kernel void sub2 (__global float* restrict output, __global float* restrict input, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) output[i] -= input[i];
}

__kernel void sub3 (__global float* restrict output, __global float* restrict input0, __global float* restrict input1, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) output[i] = input0[i] - input1[i];
}

__kernel void dsoftmax (__global float* restrict dx, __global float* restrict dy, __global float* restrict y, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) dx[i] = dy[i] - y[i];
}

__kernel void dlogistic (__global float* restrict dx, __global float* restrict dy, __global float* restrict y, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) dx[i] = dy[i] * y[i] * (1.0f - y[i]);
}

__kernel void drelu (__global float* restrict dx, __global float* restrict dy, __global float* restrict y, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) dx[i] = convert_float (y[i] > 0.0f) * dy[i];
}

void f_exp(__global float* restrict y, __global float* restrict x, const unsigned int count) {

	int i = get_global_id (0);
	if (i < count) y[i] = exp_function (x[i]);

}

__kernel void f_min_exp (__global float* restrict y, __global float* restrict x, const unsigned int count) {

	//float maxval;
	//int maxvalidx;

	//isamax(x, &maxval, &maxvalidx, count);

	//printf("f_min_exp: %f %d\n", maxval, maxvalidx);
	f_exp (y, x, count);

}

__kernel void f_softmax (__global float* restrict y, __global float* restrict x, const unsigned int count, unsigned int cols) {

	//cl_sub_max_coeff (x);
	//x -= max(x);

	//cl_softmax (y, x);
	f_exp (y, x, count);

//	if (!colsums.ref_device_data) {
//		colsums = cl_matrix<float> (y.matrix_ctx, {1, y.cols() });
//	}

	//TODO: split and improve
//	cl_colsumdiv (colsums, y, wait);

}
__kernel void colsumdiv (__global float* restrict output, __global float* restrict input, const unsigned int count, const unsigned int cols) {
	int i = get_global_id (0);
	int rows = count / cols;

	if (i < cols) {
		output[i] = 0.0f;

#pragma unroll
		for (int k = 0; k < rows; k++)  output[i] += input[k + i * rows];

#pragma unroll
		for (int k = 0; k < rows; k++)  input[k + i * rows] = divide_function(input[k + i * rows], output[i]);
	}
}

#define LOCAL_SIZE 64

__kernel void fmad_lmem (__global float* restrict p, __global float* restrict d, const float a, const unsigned int count) {

	uint tx = get_local_id(0);
	uint gx = get_global_id(0);

	__local float local_p[LOCAL_SIZE];
	__local float local_d[LOCAL_SIZE];
	__local float local_a;

	float out;

	if (gx < count) {
		// load data to local mem
		if (tx == 0) local_a = a;
		local_p[tx] = p[gx];
		local_d[tx] = d[gx];
		barrier (CLK_LOCAL_MEM_FENCE);
		out = fmad_function (local_d[tx], local_a, local_p[tx]);
		p[gx] = out;
	}
}

__kernel void fmad (__global float* restrict p, __global float* restrict d, const float a, const unsigned int count) {
	int i = get_global_id (0);

	//p[i] += d[i] * a;
	if (i < count) p[i] = fmad_function (d[i], a, p[i]);
}

__kernel void gather_data (__global float* restrict in, __global float* restrict out, __global int* restrict idxs, const unsigned int n, const unsigned int count) {
	unsigned int i = get_global_id (0);
	unsigned int k = i / n;
	unsigned int l = i % n;
	unsigned int src_idx = idxs[k] * n;
	if (i < count) out[i] = in[src_idx + l];
}

__kernel void cross_entropy (__global float* restrict logprobs, __global float* restrict pred, __global float* restrict targets, const unsigned int count) {

	int i = get_global_id (0);
	if (i < count)
		logprobs[i] = - log_function(pred[i]) * targets[i];

}
