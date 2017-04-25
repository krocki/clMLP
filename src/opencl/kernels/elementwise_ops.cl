
//#pragma OPENCL EXTENSION cl_amd_printf : enable

#define FAST_MATH

#ifdef FAST_MATH
#define exp_function native_exp
#else
#define exp_function exp
#endif

__kernel void logistic1(__global float* inputoutput, const unsigned int count) {

	int i = get_global_id(0);

	if (i < count) {
#ifdef FAST_MATH
		inputoutput[i] = native_recip ( 1.0f + exp_function ( -inputoutput[i] ) );
#else
		inputoutput[i] = 1.0f / ( 1.0f + exp ( -inputoutput[i] ) );
#endif
	}
}

__kernel void logistic2(__global float* output, __global float* input, const unsigned int count) {

	int i = get_global_id(0);

	if (i < count) {
#ifdef FAST_MATH
		output[i] = native_recip ( 1.0f + exp_function ( -input[i] ) );
#else
		output[i] = 1.0f / ( 1.0f + exp ( -input[i] ) );
#endif
	}
}

__kernel void rectify1(__global float* inputoutput, const unsigned int count) {

	int i = get_global_id(0);
	if (i < count) inputoutput[i] = fmax(inputoutput[i], 0.0f);

}

__kernel void rectify2(__global float* output, __global float* input, const unsigned int count) {

	int i = get_global_id(0);
	if (i < count) output[i] = fmax(input[i], 0.0f);

}

__kernel void expf1(__global float* inputoutput, const unsigned int count) {

	int i = get_global_id(0);
	if (i < count) {
		inputoutput[i] = exp_function(inputoutput[i]);
	}

}

__kernel void expf2(__global float* output, __global float* input, const unsigned int count) {

	int i = get_global_id(0);
	if (i < count) {
		output[i] = exp_function(input[i]);
	}
}

__kernel void sub1(__global float* output, __global float* input, const unsigned int val, const unsigned int count) {

	int i = get_global_id(0);

	if (i < count) {
		output[i] -= input[val];
	}

}

__kernel void sub2(__global float* output, __global float* input, const unsigned int count) {

	int i = get_global_id(0);
	if (i < count) output[i] -= input[i];

}

__kernel void sub3(__global float* output, __global float* input0, __global float* input1, const unsigned int count) {

	int i = get_global_id(0);
	if (i < count) output[i] = input0[i] - input1[i];

}

__kernel void dsoftmax(__global float* dx, __global float* dy, __global float* y, const unsigned int count) {

	int i = get_global_id(0);
	if (i < count) dx[i] = dy[i] - y[i];

}

__kernel void dlogistic(__global float* dx, __global float* dy, __global float* y, const unsigned int count) {

	int i = get_global_id(0);
	if (i < count) dx[i] = dy[i] * y[i] * (1.0f - y[i]);

}

__kernel void drelu(__global float* dx, __global float* dy, __global float* y, const unsigned int count) {

	int i = get_global_id(0);
	if (i < count) dx[i] = float(y[i] > 0.0f) * dy[i];

}

__kernel void colsumdiv(__global float* output, __global float* input, const unsigned int count, const unsigned int cols) {

	int i = get_global_id(0);
	int rows = count / cols;

	if (i < cols) {

		output[i] = 0.0f;

		for (int k = 0; k < rows; k++) {
			output[i] += input[k + i * rows];

		}

		for (int k = 0; k < rows; k++) {
			input[k + i * rows] /= output[i];

		}

	}

}

__kernel void fmad(__global float* p, __global float* d, const float a, const unsigned int count) {

	int i = get_global_id(0);

	if (i < count) {

		p[i] += d[i] * a;

	}

}
