/*
* @Author: kmrocki@us.ibm.com
* @Date:   2017-04-26 17:26:26
* @Last Modified by:   Kamil Rocki
* @Last Modified time: 2017-04-27 09:28:22
*/

#include <clRNG/clRNG.h>
#include <clRNG/mrg31k3p.h>

#ifndef _CL_RAND_
#define _CL_RAND_

size_t streamBufferSize = 0;
size_t numWorkItems = 64;
clrngMrg31k3pStream *streams = 0;
cl_mem bufIn;

int init_clrng ( cl_context &ctx ) {

	if ( ctx == nullptr ) {
		printf ( "init_clrng() : cl_context is null\n" );
		return 1;
	}
	
	cl_int err;
	
	streams = clrngMrg31k3pCreateStreams ( NULL, numWorkItems, &streamBufferSize, ( clrngStatus * ) &err );
	
	if ( err != CL_SUCCESS ) {
	
		printf ( "init_clrng() : clrngMrg31k3pCreateStreams failed with %d\n", err );
		return 1;
		
	}
	
	std::cout << "init_clrng: streamBufferSize = " << streamBufferSize << std::endl;
	std::cout << "init_clrng: streams = " << streams << std::endl;
	
	bufIn = clCreateBuffer ( ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, streamBufferSize, streams, &err );
	
	if ( err != CL_SUCCESS ) {
		printf ( "init_clrng() : clCreateBuffer failed with %d\n", err );
		return 1;
		
	}
	
	return 0;
}

void destroy_clrng() {

	if ( bufIn ) clReleaseMemObject ( bufIn );
	
}

#endif
