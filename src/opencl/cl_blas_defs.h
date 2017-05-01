#ifdef CLBLAST
#include <clblast_c.h>
#define CL_BLAS_SHOW_VERSION do { printf("clBlast\n"); } while (0)
#define CL_BLAS_INIT() CL_SUCCESS
#define CL_BLAS_TEARDOWN() CL_SUCCESS
#define CL_BLAS_SGEMM(order, transA, transB, M, N, K, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events ) CLBlastSgemm((order), (transA), (transB), (M), (N), (K), (alpha), (A), (offA), (lda), (B), (offB), (ldb), (beta), (C), (offC), (ldc), (commandQueues), (events))
#define CL_BLAS_ISAMAX(n, imax, imax_offset, buf, x_offset, x_inc, scratchbuf, num_queues, queue, num_events, waitlist, event) CLBlastiSamax((n), (imax), (imax_offset), (buf), (x_offset), (x_inc), (queue), (event))
#define CL_BLAS_SASUM(n, asum, offAsum, buf, x_offset, x_inc, scratchbuf, num_queues, queue, num_events, waitlist, event) CLBlastSasum((n), (asum), (offAsum), (buf), (x_offset), (x_inc), (queue), (event))
#define CL_BLAS_STATUS_TYPE CLBlastStatusCode
#define CL_BLAS_SUCCESS_CODE CLBlastSuccess
#define CL_BLAS_MATRIX_ORDER CLBlastLayout
#define CL_BLAS_MATRIX_ORDER_ROW CLBlastLayoutRowMajor
#define CL_BLAS_MATRIX_ORDER_COLUMN CLBlastLayoutColMajor
#define CL_BLAS_MATRIX_TRANSPOSE CLBlastTranspose
#define CL_BLAS_MATRIX_NOT_TRANSPOSED CLBlastTransposeNo
#define CL_BLAS_MATRIX_TRANSPOSED CLBlastTransposeYes
#endif

#ifdef CLBLAS
#include <clBLAS.h>
#define CL_BLAS_SHOW_VERSION() do { cl_uint major,minor,patch; clblasGetVersion(&major,&minor,&patch); printf("clBlas %u.%u.%u\n", major, minor, patch); } while (0)
#define CL_BLAS_INIT() clblasSetup()
#define CL_BLAS_TEARDOWN() clblasTeardown()
#define CL_BLAS_SGEMM(order, transA, transB, M, N, K, alpha, A, offA, lda, B, offB, ldb, beta, C, offC, ldc, numCommandQueues, commandQueues, numEventsInWaitList, eventWaitList, events ) clblasSgemm((order), (transA), (transB), (M), (N), (K), (alpha), (A), (offA), (lda), (B), (offB), (ldb), (beta), (C), (offC), (ldc), (numCommandQueues), (commandQueues), (numEventsInWaitList), (eventWaitList), (events))
#define CL_BLAS_ISAMAX(n, imax, imax_offset, buf, x_offset, x_inc, scratchbuf, num_queues, queue, num_events, waitlist, event) clblasiSamax((n), (imax), (imax_offset), (buf), (x_offset), (x_inc), (scratchbuf), (num_queues), (queue), (num_events), (waitlist), (event))
#define CL_BLAS_SASUM(n, asum, offAsum, buf, x_offset, x_inc, scratchbuf, num_queues, queue, num_events, waitlist, event) clblasSasum((n), (asum), (offAsum), (buf), (x_offset), (x_inc), (scratchbuf), (num_queues), (queue), (num_events), (waitlist), (event))
#define CL_BLAS_STATUS_TYPE cl_int
#define CL_BLAS_SUCCESS_CODE CL_SUCCESS
#define CL_BLAS_MATRIX_ORDER clblasOrder
#define CL_BLAS_MATRIX_ORDER_ROW clblasRowMajor
#define CL_BLAS_MATRIX_ORDER_COLUMN clblasColumnMajor
#define CL_BLAS_MATRIX_TRANSPOSE clblasTranspose
#define CL_BLAS_MATRIX_NOT_TRANSPOSED clblasNoTrans
#define CL_BLAS_MATRIX_TRANSPOSED clblasTrans
#endif