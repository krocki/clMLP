/* Sample kernel that calls clRNG device-side interfaces to generate random numbers */
#define CLRNG_SINGLE_PRECISION
#include <clRNG/mrg31k3p.clh>

__kernel void uniform01(__global clrngMrg31k3pHostStream *streams, __global float *out) {

    int gid = get_global_id(0);

    clrngMrg31k3pStream workItemStream;
    clrngMrg31k3pCopyOverStreamsFromGlobal(1, &workItemStream, &streams[gid]);

    out[gid] = clrngMrg31k3pRandomU01(&workItemStream);

}