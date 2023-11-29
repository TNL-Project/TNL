#pragma once

#ifdef __HIP__

   #include <hipblas/hipblas.h>

inline hipblasStatus_t
hipblasIgamax( hipblasHandle_t handle, int n, const float* x, int incx, int* result )
{
   return hipblasIsamax( handle, n, x, incx, result );
}

inline hipblasStatus_t
hipblasIgamax( hipblasHandle_t handle, int n, const double* x, int incx, int* result )
{
   return hipblasIdamax( handle, n, x, incx, result );
}

inline hipblasStatus_t
hipblasIgamin( hipblasHandle_t handle, int n, const float* x, int incx, int* result )
{
   return hipblasIsamin( handle, n, x, incx, result );
}

inline hipblasStatus_t
hipblasIgamin( hipblasHandle_t handle, int n, const double* x, int incx, int* result )
{
   return hipblasIdamin( handle, n, x, incx, result );
}

inline hipblasStatus_t
hipblasGasum( hipblasHandle_t handle, int n, const float* x, int incx, float* result )
{
   return hipblasSasum( handle, n, x, incx, result );
}

inline hipblasStatus_t
hipblasGasum( hipblasHandle_t handle, int n, const double* x, int incx, double* result )
{
   return hipblasDasum( handle, n, x, incx, result );
}

inline hipblasStatus_t
hipblasGaxpy( hipblasHandle_t handle, int n, const float* alpha, const float* x, int incx, float* y, int incy )
{
   return hipblasSaxpy( handle, n, alpha, x, incx, y, incy );
}

inline hipblasStatus_t
hipblasGaxpy( hipblasHandle_t handle, int n, const double* alpha, const double* x, int incx, double* y, int incy )
{
   return hipblasDaxpy( handle, n, alpha, x, incx, y, incy );
}

inline hipblasStatus_t
hipblasGdot( hipblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result )
{
   return hipblasSdot( handle, n, x, incx, y, incy, result );
}

inline hipblasStatus_t
hipblasGdot( hipblasHandle_t handle, int n, const double* x, int incx, const double* y, int incy, double* result )
{
   return hipblasDdot( handle, n, x, incx, y, incy, result );
}

inline hipblasStatus_t
hipblasGnrm2( hipblasHandle_t handle, int n, const float* x, int incx, float* result )
{
   return hipblasSnrm2( handle, n, x, incx, result );
}

inline hipblasStatus_t
hipblasGnrm2( hipblasHandle_t handle, int n, const double* x, int incx, double* result )
{
   return hipblasDnrm2( handle, n, x, incx, result );
}

inline hipblasStatus_t
hipblasGscal( hipblasHandle_t handle, int n, const float* alpha, float* x, int incx )
{
   return hipblasSscal( handle, n, alpha, x, incx );
}

inline hipblasStatus_t
hipblasGscal( hipblasHandle_t handle, int n, const double* alpha, double* x, int incx )
{
   return hipblasDscal( handle, n, alpha, x, incx );
}

inline hipblasStatus_t
hipblasGemv( hipblasHandle_t handle,
             hipblasOperation_t trans,
             int m,
             int n,
             const float* alpha,
             const float* A,
             int lda,
             const float* x,
             int incx,
             const float* beta,
             float* y,
             int incy )
{
   return hipblasSgemv( handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy );
}

inline hipblasStatus_t
hipblasGemv( hipblasHandle_t handle,
             hipblasOperation_t trans,
             int m,
             int n,
             const double* alpha,
             const double* A,
             int lda,
             const double* x,
             int incx,
             const double* beta,
             double* y,
             int incy )
{
   return hipblasDgemv( handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy );
}

#endif
