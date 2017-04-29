/* Sample kernel that calls clRNG device-side interfaces to generate random numbers */
#define CLRNG_SINGLE_PRECISION
#include <clRNG/mrg31k3p.clh>
#include <clRNG/mrg32k3a.clh>

#define CLPROBDIST_NORMAL_OBJ_MEM CLPROBDIST_MEM_TYPE_PRIVATE
//#define CLRNG_ENABLE_SUBSTREAMS

//prob dist
#include <clProbDist/normal.clh>

__kernel void uniform01 (__global clrngMrg31k3pHostStream* streams, __global float* out, const unsigned int iters, const unsigned int count) {
	unsigned int gid = get_global_id (0);
	unsigned int gsize = get_global_size (0);
	unsigned int idx;
	clrngMrg31k3pStream workItemStream;
	clrngMrg31k3pCopyOverStreamsFromGlobal (1, &workItemStream, &streams[gid]);

#pragma unroll
	for (unsigned int i = 0; i < iters; i++) {
		idx = gid + gsize * i;
		if (idx < count) out[gid + gsize * i] = clrngMrg31k3pRandomU01 (&workItemStream);
	}

	//write the state back
	clrngMrg31k3pCopyOverStreamsToGlobal (1, &streams[gid], &workItemStream);
}

__kernel void randi (__global clrngMrg31k3pHostStream* streams, __global int* out, const int r_min, const int r_max, const unsigned int iters, const unsigned int count) {
	unsigned int gid = get_global_id (0);
	unsigned int gsize = get_global_size (0);
	unsigned int idx;
	clrngMrg31k3pStream* workItemStream;
	clrngMrg31k3pCopyOverStreamsFromGlobal (1, &workItemStream, &streams[gid]);

#pragma unroll
	for (unsigned int i = 0; i < iters; i++) {
		idx = gid + gsize * i;
		if (idx < count) out[gid + gsize * i] = clrngMrg31k3pRandomInteger (&workItemStream, r_min, r_max);
	}

	//write the state back
	clrngMrg31k3pCopyOverStreamsToGlobal (1, &streams[gid], &workItemStream);
}

// TODO: apparently instead of iters, can run multiple streams
__kernel void normal (__global clrngMrg31k3pHostStream* streams, __global clprobdistNormal* g_normalDist, __global float* out, const unsigned int iters, const unsigned int count) {
	int gid = get_global_id (0);   // Id of this work item.
	int gsize = get_global_size (0);   // Total number of work items
	unsigned int idx;
	clprobdistStatus err;
	// Make copies of the stream states in private memory.
	clrngMrg31k3pStream workItemStream;
	clrngMrg31k3pCopyOverStreamsFromGlobal (1, &workItemStream, &streams[gid]);
	//Make a copy of clprobdistNormal in private memory
	clprobdistNormal p_normalDist = *g_normalDist;

#pragma unroll
	for (unsigned int i = 0; i < iters; i++) {
		idx = gid + gsize * i;
		if (idx < count) {
			float u = clrngMrg31k3pRandomU01 (&workItemStream);
			out[i * gsize + gid] = convert_float (clprobdistNormalInverseCDFWithObject (&p_normalDist, u, &err) );
		}
	}

	//write the state back
	clrngMrg31k3pCopyOverStreamsToGlobal (1, &streams[gid], &workItemStream);
}
