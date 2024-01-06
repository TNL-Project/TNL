// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/*
 * TODO: This is just a temporary file, used only in the CWYGMRES solver.
 * The algorithms should be incorporated into the Matrices::Dense class.
 */

#include <memory>  // std::unique_ptr

#include <TNL/Backend.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/copy.h>
#include <TNL/Containers/Vector.h>

namespace TNL::Matrices {

template< typename DeviceType = Devices::Host >
class MatrixOperations
{
public:
   /*
    * This function performs the matrix-vector multiplication
    *    y = alpha * A * x + beta * y
    * where:
    *    alpha and beta are scalars,
    *    A is an (lda by n) matrix stored in column-major format,
    *    lda >= m is the leading dimension of two-dimensional array used to store matrix A,
    *    x is a vector of n elements,
    *    y is a vector of m elements.
    *
    * It is assumed that n is much smaller than m.
    */
   template< typename RealType, typename IndexType >
   static void
   gemv( const IndexType m,
         const IndexType n,
         const RealType alpha,
         const RealType* A,
         const IndexType lda,
         const RealType* x,
         const RealType beta,
         RealType* y )
   {
      if( m <= 0 )
         throw std::invalid_argument( "gemv: m must be positive" );
      if( n <= 0 )
         throw std::invalid_argument( "gemv: n must be positive" );
      if( lda < m )
         throw std::invalid_argument( "gemv: lda must be at least m" );

      std::unique_ptr< RealType[] > alphax{ new RealType[ n ] };
      for( IndexType k = 0; k < n; k++ )
         alphax[ k ] = alpha * x[ k ];

      if( n == 1 ) {
         if( beta != 0.0 ) {
#ifdef HAVE_OPENMP
            #pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() )
#endif
            for( IndexType j = 0; j < m; j++ )
               y[ j ] = A[ j ] * alphax[ 0 ] + beta * y[ j ];
         }
         else {
// the vector y might be uninitialized, and 0.0 * NaN = NaN
#ifdef HAVE_OPENMP
            #pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() )
#endif
            for( IndexType j = 0; j < m; j++ )
               y[ j ] = A[ j ] * alphax[ 0 ];
         }
      }
      else {
         // the matrix A should be accessed column-wise so we split the work into small
         // blocks and each block process by columns, either parallelly or serially
         constexpr IndexType block_size = 128;
         const IndexType blocks = m / block_size;

#ifdef HAVE_OPENMP
         #pragma omp parallel if( TNL::Devices::Host::isOMPEnabled() && blocks >= 2 )
#endif
         {
            RealType aux[ block_size ];

#ifdef HAVE_OPENMP
            #pragma omp for nowait
#endif
            for( IndexType b = 0; b < blocks; b++ ) {
               const IndexType block_offset = b * block_size;

               // initialize array for thread-local results
               for( IndexType j = 0; j < block_size; j++ )
                  aux[ j ] = 0.0;

               // compute aux = A * alphax
               for( IndexType k = 0; k < n; k++ ) {
                  const IndexType offset = block_offset + k * lda;
                  for( IndexType j = 0; j < block_size; j++ )
                     aux[ j ] += A[ offset + j ] * alphax[ k ];
               }

               // write result: y = aux + beta * y
               if( beta != 0.0 ) {
                  for( IndexType j = 0; j < block_size; j++ )
                     y[ block_offset + j ] = aux[ j ] + beta * y[ block_offset + j ];
               }
               else {
                  // the vector y might be uninitialized, and 0.0 * NaN = NaN
                  for( IndexType j = 0; j < block_size; j++ )
                     y[ block_offset + j ] = aux[ j ];
               }
            }

// the first thread that reaches here processes the last, incomplete block
#ifdef HAVE_OPENMP
            #pragma omp single nowait
#endif
            {
               // TODO: unlike the complete blocks, the tail is traversed row-wise
               if( beta != 0.0 ) {
                  for( IndexType j = blocks * block_size; j < m; j++ ) {
                     RealType tmp = 0.0;
                     for( IndexType k = 0; k < n; k++ )
                        tmp += A[ j + k * lda ] * alphax[ k ];
                     y[ j ] = tmp + beta * y[ j ];
                  }
               }
               else {
                  // the vector y might be uninitialized, and 0.0 * NaN = NaN
                  for( IndexType j = blocks * block_size; j < m; j++ ) {
                     RealType tmp = 0.0;
                     for( IndexType k = 0; k < n; k++ )
                        tmp += A[ j + k * lda ] * alphax[ k ];
                     y[ j ] = tmp;
                  }
               }
            }
         }
      }
   }

   /*
    * This function performs the matrix-matrix addition
    *    C = alpha * A + beta * B
    * where:
    *    alpha and beta are scalars,
    *    A, B, C are (m by n) matrices stored in column-major format on Devices::Cuda,
    *    lda, ldb, ldc (all >= m) are the leading dimensions of matrices A, B, C,
    *    respectively.
    *
    * It is assumed that n is much smaller than m.
    */
   template< typename RealType, typename IndexType >
   static void
   geam( const IndexType m,
         const IndexType n,
         const RealType alpha,
         const RealType* A,
         const IndexType lda,
         const RealType beta,
         const RealType* B,
         const IndexType ldb,
         RealType* C,
         const IndexType ldc )
   {
      if( m <= 0 )
         throw std::invalid_argument( "geam: m must be positive" );
      if( n <= 0 )
         throw std::invalid_argument( "geam: n must be positive" );
      if( lda < m )
         throw std::invalid_argument( "geam: lda must be at least m" );
      if( ldb < m )
         throw std::invalid_argument( "geam: ldb must be at least m" );
      if( ldc < m )
         throw std::invalid_argument( "geam: ldc must be at least m" );

      if( n == 1 ) {
#ifdef HAVE_OPENMP
         #pragma omp parallel for if( TNL::Devices::Host::isOMPEnabled() )
#endif
         for( IndexType j = 0; j < m; j++ )
            C[ j ] = alpha * A[ j ] + beta * B[ j ];
      }
      else {
         // all matrices should be accessed column-wise so we split the work into small
         // blocks and each block process by columns, either parallelly or serially
         constexpr IndexType block_size = 128;
         const IndexType blocks = m / block_size;

#ifdef HAVE_OPENMP
         #pragma omp parallel if( TNL::Devices::Host::isOMPEnabled() && blocks >= 2 )
#endif
         {
#ifdef HAVE_OPENMP
            #pragma omp for nowait
#endif
            for( IndexType b = 0; b < blocks; b++ ) {
               const IndexType block_offset = b * block_size;
               for( IndexType j = 0; j < n; j++ ) {
                  const IndexType offset_A = j * lda + block_offset;
                  const IndexType offset_B = j * ldb + block_offset;
                  const IndexType offset_C = j * ldc + block_offset;
                  for( IndexType i = 0; i < block_size; i++ )
                     C[ offset_C + i ] = alpha * A[ offset_A + i ] + beta * B[ offset_B + i ];
               }
            }

// the first thread that reaches here processes the last, incomplete block
#ifdef HAVE_OPENMP
            #pragma omp single nowait
#endif
            {
               for( IndexType j = 0; j < n; j++ ) {
                  const IndexType offset_A = j * lda;
                  const IndexType offset_B = j * ldb;
                  const IndexType offset_C = j * ldc;
                  for( IndexType i = blocks * block_size; i < m; i++ )
                     C[ offset_C + i ] = alpha * A[ offset_A + i ] + beta * B[ offset_B + i ];
               }
            }
         }
      }
   }
};

// CUDA kernels
template< typename RealType, typename IndexType >
__global__
void
GemvCudaKernel( const IndexType m,
                const IndexType n,
                const RealType alpha,
                const RealType* A,
                const IndexType lda,
                const RealType* x,
                const RealType beta,
                RealType* y )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   IndexType elementIdx = blockIdx.x * blockDim.x + threadIdx.x;
   const IndexType gridSize = blockDim.x * gridDim.x;

   RealType* shx = Backend::getSharedMemory< RealType >();

   if( threadIdx.x < n )
      shx[ threadIdx.x ] = alpha * x[ threadIdx.x ];
   __syncthreads();

   if( beta != 0.0 ) {
      while( elementIdx < m ) {
         RealType tmp = 0.0;
         for( IndexType k = 0; k < n; k++ )
            tmp += A[ elementIdx + k * lda ] * shx[ k ];
         y[ elementIdx ] = tmp + beta * y[ elementIdx ];
         elementIdx += gridSize;
      }
   }
   else {
      // the vector y might be uninitialized, and 0.0 * NaN = NaN
      while( elementIdx < m ) {
         RealType tmp = 0.0;
         for( IndexType k = 0; k < n; k++ )
            tmp += A[ elementIdx + k * lda ] * shx[ k ];
         y[ elementIdx ] = tmp;
         elementIdx += gridSize;
      }
   }
#endif
}

template< typename RealType, typename IndexType >
__global__
void
GeamCudaKernel( const IndexType m,
                const IndexType n,
                const RealType alpha,
                const RealType* A,
                const IndexType lda,
                const RealType beta,
                const RealType* B,
                const IndexType ldb,
                RealType* C,
                const IndexType ldc )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   IndexType x = blockIdx.x * blockDim.x + threadIdx.x;
   const IndexType gridSizeX = blockDim.x * gridDim.x;
   const IndexType y = blockIdx.y * blockDim.y + threadIdx.y;
   const IndexType offset_A = y * lda;
   const IndexType offset_B = y * ldb;
   const IndexType offset_C = y * ldc;

   if( y < n )
      while( x < m ) {
         C[ x + offset_C ] = alpha * A[ x + offset_A ] + beta * B[ x + offset_B ];
         x += gridSizeX;
      }
#endif
}

// specialization for CUDA
template<>
class MatrixOperations< Devices::Cuda >
{
public:
   /*
    * This function performs the matrix-vector multiplication
    *    y = alpha * A * x + beta * y
    * where:
    *    alpha and beta are scalars,
    *    A is an (lda by n) matrix stored in column-major format on Devices::Cuda,
    *    lda >= m is the leading dimension of two-dimensional array used to store matrix A,
    *    x is a vector of n elements, stored on Devices::Host,
    *    y is a vector of m elements, stored on Devices::Cuda.
    *
    * It is assumed that n is much smaller than m.
    */
   template< typename RealType, typename IndexType >
   static void
   gemv( const IndexType m,
         const IndexType n,
         const RealType alpha,
         const RealType* A,
         const IndexType lda,
         const RealType* x,
         const RealType beta,
         RealType* y )
   {
      if( m > lda )
         throw std::invalid_argument( "gemv: the size 'm' must be less than or equal to 'lda'." );
      if( n > 256 )
         throw std::invalid_argument( "The gemv kernel is optimized only for small 'n' and assumes that n <= 256." );

      // TODO: use static storage, e.g. from the CudaReductionBuffer, to avoid frequent reallocations
      Containers::Vector< RealType, Devices::Cuda, IndexType > xDevice;
      xDevice.setSize( n );
      Algorithms::copy< Devices::Cuda, Devices::Host >( xDevice.getData(), x, n );

      // desGridSize = blocksPerMultiprocessor * numberOfMultiprocessors
      const int desGridSize = 32 * Backend::getDeviceMultiprocessors( Backend::getDevice() );
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = 256;
      launch_config.gridSize.x = min( desGridSize, Backend::getNumberOfBlocks( m, launch_config.blockSize.x ) );
      launch_config.dynamicSharedMemorySize = n * sizeof( RealType );

      constexpr auto kernel = GemvCudaKernel< RealType, IndexType >;
      Backend::launchKernelSync( kernel, launch_config, m, n, alpha, A, lda, xDevice.getData(), beta, y );
   }

   /*
    * This function performs the matrix-matrix addition
    *    C = alpha * A + beta * B
    * where:
    *    alpha and beta are scalars,
    *    A, B, C are (m by n) matrices stored in column-major format on Devices::Cuda,
    *    lda, ldb, ldc (all >= m) are the leading dimensions of matrices A, B, C,
    *    respectively.
    *
    * It is assumed that n is much smaller than m.
    */
   template< typename RealType, typename IndexType >
   static void
   geam( const IndexType m,
         const IndexType n,
         const RealType alpha,
         const RealType* A,
         const IndexType lda,
         const RealType beta,
         const RealType* B,
         const IndexType ldb,
         RealType* C,
         const IndexType ldc )
   {
      if( m <= 0 )
         throw std::invalid_argument( "geam: m must be positive" );
      if( n <= 0 )
         throw std::invalid_argument( "geam: n must be positive" );
      if( lda < m )
         throw std::invalid_argument( "geam: lda must be at least m" );
      if( ldb < m )
         throw std::invalid_argument( "geam: ldb must be at least m" );
      if( ldc < m )
         throw std::invalid_argument( "geam: ldc must be at least m" );

      Backend::LaunchConfiguration launch_config;

      // max 16 columns of threads
      launch_config.blockSize.y = min( n, 16 );
      // max 256 threads per block, power of 2
      launch_config.blockSize.x = 256;
      while( launch_config.blockSize.x * launch_config.blockSize.y > 256 )
         launch_config.blockSize.x /= 2;

      // desGridSize = blocksPerMultiprocessor * numberOfMultiprocessors
      const int desGridSize = 32 * Backend::getDeviceMultiprocessors( Backend::getDevice() );
      launch_config.gridSize.x = min( desGridSize, Backend::getNumberOfBlocks( m, launch_config.blockSize.x ) );
      launch_config.gridSize.y = Backend::getNumberOfBlocks( n, launch_config.blockSize.y );

      constexpr auto kernel = GeamCudaKernel< RealType, IndexType >;
      Backend::launchKernelSync( kernel, launch_config, m, n, alpha, A, lda, beta, B, ldb, C, ldc );
   }
};

/**
 * \brief This function computes \f$( A + A^T ) / 2 \f$, where \f$ A \f$ is a square matrix.
 *
 * \tparam InMatrix is the type of the input matrix.
 * \tparam OutMatrix is the type of the output matrix.
 * \param inMatrix is the input matrix.
 * \return the output matrix.
 */
template< typename OutMatrix, typename InMatrix >
OutMatrix
getSymmetricPart( const InMatrix& inMatrix )
{
   static_assert(
      std::is_same_v< typename InMatrix::DeviceType, Devices::Host >
         || std::is_same_v< typename InMatrix::DeviceType, Devices::Sequential >,
      "The input matrix must be stored on host, i.e. only Devices::Host and Devices::Sequential devices are allowed." );
   if( inMatrix.getRows() != inMatrix.getColumns() )
      throw std::invalid_argument( "getSymmetricPart: the input matrix must be square" );

   // TODO: the following needs to be optimized and it works only for sparse matrices on host
   using RealType = typename InMatrix::RealType;
   using IndexType = typename InMatrix::IndexType;

   OutMatrix outMatrix;
   std::map< std::pair< IndexType, IndexType >, RealType > map;
   for( IndexType rowIdx = 0; rowIdx < inMatrix.getRows(); rowIdx++ ) {
      auto row = inMatrix.getRow( rowIdx );
      for( IndexType localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
         IndexType columnIdx = row.getColumnIndex( localIdx );
         RealType value = row.getValue( localIdx );
         if( auto element = map.find( std::make_pair( rowIdx, columnIdx ) ); element != map.end() )
            value = ( value + element->second ) / 2.0;
         map[ std::make_pair( rowIdx, columnIdx ) ] = value;
         map[ std::make_pair( columnIdx, rowIdx ) ] = value;
      }
   }
   outMatrix.setDimensions( inMatrix.getRows(), inMatrix.getColumns() );
   outMatrix.setElements( map );
   return outMatrix;
}

}  // namespace TNL::Matrices
