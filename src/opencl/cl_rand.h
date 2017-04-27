/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-04-26 17:26:26
* @Last Modified by:   kmrocki@us.ibm.com
* @Last Modified time: 2017-04-27 15:27:31
*/

#include <clRNG/clRNG.h>
#include <clRNG/mrg31k3p.h>
#include <clRNG/mrg32k3a.h>

#include <clProbDist/clProbDist.h>
#include <clProbDist/exponential.h>
#include <clProbDist/normal.h>

#ifndef _CL_RAND_
#define _CL_RAND_

size_t numWorkItems = 64;

size_t streamUniformBufferSize = 0;
size_t streamNormalBufferSize = 0;
clrngMrg31k3pStream *streamsUniform = 0;
// clrngMrg32k3aStream *streamsNormal = 0;
clrngMrg31k3pStream *streamsNormal = 0;
cl_mem bufInUniform, bufInNormal, normalDist_buffer;

int init_clrng ( cl_context &ctx, size_t workitems = 64 ) {

	if ( ctx == nullptr ) {
		printf ( "init_clrng() : cl_context is null\n" );
		return 1;
	}

	cl_int err;

	numWorkItems = workitems;

	streamsUniform = clrngMrg31k3pCreateStreams ( NULL, numWorkItems, &streamUniformBufferSize, ( clrngStatus * ) &err );

	if ( err != CL_SUCCESS ) {

		printf ( "init_clrng() : streamsUniform failed with %d\n", err );
		return 1;

	}

	// streamsNormal = clrngMrg32k3aCreateStreams ( NULL, numWorkItems, &streamNormalBufferSize, ( clrngStatus * ) &err );
	streamsNormal = clrngMrg31k3pCreateStreams ( NULL, numWorkItems, &streamNormalBufferSize, ( clrngStatus * ) &err );

	if ( err != CL_SUCCESS ) {

		printf ( "init_clrng() : streamsNormal failed with %d\n", err );
		return 1;

	}

	std::cout << "init_clrng: streamNormalBufferSize = " << streamNormalBufferSize << std::endl;
	std::cout << "init_clrng: streamUniformBufferSize = " << streamUniformBufferSize << std::endl;
	std::cout << "init_clrng: streamsUniform = " << streamsUniform << std::endl;
	std::cout << "init_clrng: streamsNormal = " << streamsNormal << std::endl;

	bufInNormal = clCreateBuffer ( ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, streamNormalBufferSize, streamsNormal, &err );
	bufInUniform = clCreateBuffer ( ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, streamUniformBufferSize, streamsUniform, &err );

	//set normal distribution
	size_t normalBufSize = 0;
	float mu = 0.0f;
	float sigma = 1.0f;

	clprobdistNormal* normalDist = clprobdistNormalCreate(mu, sigma, &normalBufSize, (clprobdistStatus *) &err);
	if ( err != CL_SUCCESS ) printf ( "cl_matrix_randn : clprobdistNormalCreate failed with %d\n", err );

	std::cout << "normalBufSize " << normalBufSize << std::endl;

	normalDist_buffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, normalBufSize, normalDist, &err);
	check_error(err, "%s(): cannot create normalDist_buffer buffer", __func__);

	if ( err != CL_SUCCESS ) {
		printf ( "init_clrng() : clCreateBuffer failed with %d\n", err );
		return 1;

	}

	return 0;
}

void destroy_clrng() {

	if ( bufInNormal ) clReleaseMemObject ( bufInNormal );
	if ( bufInUniform ) clReleaseMemObject ( bufInUniform );
	if ( normalDist_buffer ) clReleaseMemObject(normalDist_buffer);

}

#endif
