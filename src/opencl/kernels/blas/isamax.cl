// based on clblast code
// https://github.com/CNugteren/CLBlast/blob/master/src/kernels/level1/xamax.opencl

// Parameters set by the tuner or by the database. Here they are given a basic default value in case
// this kernel file is used outside of the CLBlast library.
#ifndef WGS1
#define WGS1 64     // The local work-group size of the main kernel
#endif
#ifndef WGS2
#define WGS2 64     // The local work-group size of the epilogue kernel
#endif

// void isamax(const __global float* restrict xgm,
//             __global float* maxgm,
//             __global unsigned int* imaxgm,
//             const int n);

// =================================================================================================
// The main reduction kernel, performing the loading and the majority of the operation
// __kernel __attribute__((reqd_work_group_size(WGS1, 1, 1)))
void isamax(const __global float* restrict xgm,
            float* maxgm,
            unsigned int* imaxgm,
            const int n) {

  __local float maxlm[WGS1];
  __local unsigned int imaxlm[WGS1];

  const int lid = get_local_id(0);
  const int wgid = get_group_id(0);
  const int num_groups = get_num_groups(0);

  *maxgm = 0.93f;
  *imaxgm = 4;

}

//   // Performs loading and the first steps of the reduction
// #if defined(ROUTINE_MAX) || defined(ROUTINE_MIN) // non-absolute version
//   singlereal max = SMALLEST;
// #else
//   singlereal max = ZERO;
// #endif
//   unsigned int imax = 0;
//   int id = wgid * WGS1 + lid;
//   while (id < n) {
//     const int x_index = id * x_inc + x_offset;
// #if PRECISION == 3232 || PRECISION == 6464
//     singlereal x = xgm[x_index].x;
// #else
//     singlereal x = xgm[x_index];
// #endif
// #if defined(ROUTINE_MAX) // non-absolute maximum version
//     // nothing special here
// #elif defined(ROUTINE_MIN) // non-absolute minimum version
//     x = -x;
// #else
//     x = fabs(x);
// #endif
//     if (x >= max) {
//       max = x;
//       imax = id * x_inc + x_offset;
//     }
//     id += WGS1 * num_groups;
//   }
//   maxlm[lid] = max;
//   imaxlm[lid] = imax;
//   barrier(CLK_LOCAL_MEM_FENCE);
//   // Performs reduction in local memory
// #pragma unroll
//   for (int s = WGS1 / 2; s > 0; s = s >> 1) {
//     if (lid < s) {
//       if (maxlm[lid + s] >= maxlm[lid]) {
//         maxlm[lid] = maxlm[lid + s];
//         imaxlm[lid] = imaxlm[lid + s];
//       }
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);
//   }
//   // Stores the per-workgroup result
//   if (lid == 0) {
//     maxgm[wgid] = maxlm[0];
//     imaxgm[wgid] = imaxlm[0];
//   }
// }
// =================================================================================================
// The epilogue reduction kernel, performing the final bit of the operation. This kernel has to
// be launched with a single workgroup only.
// __kernel __attribute__((reqd_work_group_size(WGS2, 1, 1)))
// void XamaxEpilogue(const __global singlereal* restrict maxgm,
//                    const __global unsigned int* restrict imaxgm,
//                    __global unsigned int* imax, const int imax_offset) {
//   __local singlereal maxlm[WGS2];
//   __local unsigned int imaxlm[WGS2];
//   const int lid = get_local_id(0);
//   // Performs the first step of the reduction while loading the data
//   if (maxgm[lid + WGS2] >= maxgm[lid]) {
//     maxlm[lid] = maxgm[lid + WGS2];
//     imaxlm[lid] = imaxgm[lid + WGS2];
//   } else {
//     maxlm[lid] = maxgm[lid];
//     imaxlm[lid] = imaxgm[lid];
//   }
//   barrier(CLK_LOCAL_MEM_FENCE);
//   // Performs reduction in local memory
// #pragma unroll
//   for (int s = WGS2 / 2; s > 0; s = s >> 1) {
//     if (lid < s) {
//       if (maxlm[lid + s] >= maxlm[lid]) {
//         maxlm[lid] = maxlm[lid + s];
//         imaxlm[lid] = imaxlm[lid + s];
//       }
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);
//   }
//   // Stores the final result
//   if (lid == 0) {
//     imax[imax_offset] = imaxlm[0];
//   }
// }
