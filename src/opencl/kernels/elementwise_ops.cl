
// #pragma OPENCL EXTENSION cl_amd_printf : enable

#define FAST_MATH

#ifdef FAST_MATH
#define exp_function native_exp
#define fmad_function(a,b,c) (mad ((a), (b), (c)))
#define logistic_function(x) (native_recip ( 1.0f + native_exp ( -(x) ) ))
#else
#define exp_function exp
#define fmad_function(a,b,c) ((c) + (a) * (b))
#define logistic_function(x) (1.0f / ( 1.0f + exp ( -(x) ) ))
#endif

__kernel void logistic1 (__global float* inputoutput, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) inputoutput[i] = logistic_function (inputoutput[i]);
}

__kernel void logistic2 (__global float* output, __global float* input, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) output[i] = logistic_function (input[i]);
}

__kernel void rectify1 (__global float* inputoutput, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) inputoutput[i] = fmax (inputoutput[i], 0.0f);
}

__kernel void rectify2 (__global float* output, __global float* input, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) output[i] = fmax (input[i], 0.0f);
}

__kernel void expf1 (__global float* inputoutput, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) inputoutput[i] = exp_function (inputoutput[i]);
}

__kernel void expf2 (__global float* output, __global float* input, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) output[i] = exp_function (input[i]);
}

__kernel void sub1 (__global float* output, __global float* input, const unsigned int val, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) output[i] -= input[val];
}

__kernel void sub2 (__global float* output, __global float* input, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) output[i] -= input[i];
}

__kernel void sub3 (__global float* output, __global float* input0, __global float* input1, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) output[i] = input0[i] - input1[i];
}

__kernel void dsoftmax (__global float* dx, __global float* dy, __global float* y, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) dx[i] = dy[i] - y[i];
}

__kernel void dlogistic (__global float* dx, __global float* dy, __global float* y, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) dx[i] = dy[i] * y[i] * (1.0f - y[i]);
}

__kernel void drelu (__global float* dx, __global float* dy, __global float* y, const unsigned int count) {
	int i = get_global_id (0);
	if (i < count) dx[i] = convert_float (y[i] > 0.0f) * dy[i];
}

__kernel void colsumdiv (__global float* output, __global float* input, const unsigned int count, const unsigned int cols) {
	int i = get_global_id (0);
	int rows = count / cols;

	if (i < cols) {
		output[i] = 0.0f;

#pragma unroll
		for (int k = 0; k < rows; k++)  output[i] += input[k + i * rows];

#pragma unroll
		for (int k = 0; k < rows; k++)  input[k + i * rows] /= output[i];
	}
}

#define LOCAL_SIZE 64

__kernel void fmad_lmem (__global float* p, __global float* d, const float a, const unsigned int count) {
	int i = get_global_id (0);
	int l = get_local_id (0);
	int g = get_group_id (0);
	__local float local_p[LOCAL_SIZE];
	__local float local_d[LOCAL_SIZE];
	__local float local_a;

	if (i < count) {
		// load data to local mem
		local_a = a;
		local_p[l] = p[g * LOCAL_SIZE + l];
		local_d[l] = d[g * LOCAL_SIZE + l];
		barrier (CLK_LOCAL_MEM_FENCE);
		local_p[l] = fmad_function (local_d[l], local_a, local_p[l]);
		p[g * LOCAL_SIZE + l] = local_p[l];
	}
}

__kernel void fmad (__global float* p, __global float* d, const float a, const unsigned int count) {
	int i = get_global_id (0);

	//p[i] += d[i] * a;
	if (i < count) p[i] = fmad_function (d[i], a, p[i]);
}

__kernel void gather_data (__global float* in, __global float* out, __global int* idxs, const unsigned int n, const unsigned int count) {
	unsigned int i = get_global_id (0);
	unsigned int k = i / n;
	unsigned int l = i % n;
	unsigned int src_idx = idxs[k] * n;
	if (i < count) out[i] = in[src_idx + l];
}

__kernel void cross_entropy (__global float* logprobs, __global float* pred, __global float* targets, const unsigned int count) {

	int i = get_global_id (0);
	if (i < count)
		logprobs[i] = - log(pred[i]) * targets[i];

}
