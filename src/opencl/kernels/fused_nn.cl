// #pragma OPENCL EXTENSION cl_amd_printf : enable

void f_sgemm(__global float* restrict y, __global float* restrict W, __global float* x, const unsigned int count);
void f_relu(__global float* restrict y, __global float* restrict x, const unsigned int count);
void f_softmax(__global float* restrict y, __global float* restrict x, const unsigned int count);

void f_sgemm(__global float* restrict y, __global float* restrict W, __global float* restrict x, const unsigned int count) {


}

void f_relu(__global float* restrict y, __global float* restrict x, const unsigned int count) {

	uint i = get_global_id (0);
	if (i < count) y[i] = fmax (x[i], 0.0f);
}

void f_softmax(__global float* restrict y, __global float* restrict x, const unsigned int count) {

	uint i = get_global_id (0);
	if (i < count) y[i] = x[i];

}

__kernel void forward (	__global float* restrict y0, __global float* restrict W0, __global float* restrict x0,
                        __global float* restrict y1, __global float* restrict x1,
                        __global float* restrict y2, __global float* restrict W2, __global float* restrict x2,
                        __global float* restrict y3, __global float* restrict x3,
                        __global float* restrict y4, __global float* restrict W4, __global float* restrict x4,
                        __global float* restrict y5, __global float* restrict x5,
                        const unsigned int count

                      ) {

	f_sgemm(y0, W0, x0, count);
	f_relu(y1, y0, count);
	f_sgemm(y2, W2, y1, count);
	f_relu(y3, y2, count);
	f_sgemm(y4, W4, y3, count);
	f_softmax(y5, y4, count);

}

// 1. sgemm(y0, W0, x0);
// 2. relu(y1, y0);
// 3. sgemm(y2, W2, y1);
// 4. relu(y3, y2);
// 5. sgemm(y4, W4, y3);
// 6. softmax(y5, y4);

// nn.layers.push_back (new Linear (&ocl, 28 * 28, 256, batch_size) );
// nn.layers.push_back (new ReLU (&ocl, 256, 256, batch_size) );
// nn.layers.push_back (new Linear (&ocl, 256, 100, batch_size) );
// nn.layers.push_back (new ReLU (&ocl, 100, 100, batch_size) );
// nn.layers.push_back (new Linear (&ocl, 100, 10, batch_size) );
// nn.layers.push_back (new Softmax (&ocl, 10, 10, batch_size) );