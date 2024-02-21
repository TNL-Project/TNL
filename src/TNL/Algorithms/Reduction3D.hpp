// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>  // std::unique_ptr

//#define CUDA_REDUCTION_PROFILING

#include <TNL/Assert.h>
#include <TNL/Algorithms/Reduction3D.h>
#include <TNL/Algorithms/copy.h>
#include <TNL/Algorithms/detail/CudaReduction3DKernel.h>

#ifdef CUDA_REDUCTION_PROFILING
   #include <TNL/Timer.h>
   #include <iostream>
#endif

namespace TNL::Algorithms {

template< typename Result, typename DataFetcher, typename Reduction, typename Index, typename Output >
void constexpr Reduction3D< Devices::Sequential >::reduce( Result identity,
                                                           DataFetcher dataFetcher,
                                                           Reduction reduction,
                                                           Index size,
                                                           int m,
                                                           int n,
                                                           Output result )
{
   TNL_ASSERT_GT( size, 0, "The size of datasets must be positive." );
   TNL_ASSERT_GT( m, 0, "The number of datasets must be positive." );
   TNL_ASSERT_GT( n, 0, "The number of datasets must be positive." );

   constexpr int block_size = 128;
   const int blocks = size / block_size;

#if defined( __CUDA_ARCH__ )
   for( int i = 0; i < m; i++ ) {
      for( int j = 0; j < n; j++ ) {
         result( i, j ) = identity;
      }
   }

   for( int b = 0; b < blocks; b++ ) {
      const Index offset = b * block_size;
      for( int i = 0; i < m; i++ ) {
         for( int j = 0; j < n; j++ ) {
            for( int k = 0; k < block_size; k++ ) {
               result( i, j ) = reduction( result( i, j ), dataFetcher( offset + k, i, j ) );
            }
         }
      }
   }

   for( int i = 0; i < m; i++ ) {
      for( int j = 0; j < n; j++ ) {
         for( int k = blocks * block_size; k < size; k++ ) {
            result( i, j ) = reduction( result( i, j ), dataFetcher( k, i, j ) );
         }
      }
   }
#else
   if( blocks > 1 ) {
      // initialize array for unrolled results
      // (it is accessed as a row-major matrix with n X-Axis, m Y-Axis and 4 Z-Axis)

      std::unique_ptr< Result[] > r{ new Result[ m * n * 4 ] };
      for( int i = 0; i < m * n * 4; i++ )
         r[ i ] = identity;

      // main reduction (explicitly unrolled loop)
      for( int b = 0; b < blocks; b++ ) {
         const Index offset = b * block_size;
         for( int i = 0; i < m; i++ ) {
            for( int j = 0; j < n; j++ ) {
               Result* _r = r.get() + ( i * n + j ) * 4;
               for( int k = 0; k < block_size; k += 4 ) {
                  _r[ 0 ] = reduction( _r[ 0 ], dataFetcher( offset + k, i, j ) );
                  _r[ 1 ] = reduction( _r[ 1 ], dataFetcher( offset + k + 1, i, j ) );
                  _r[ 2 ] = reduction( _r[ 2 ], dataFetcher( offset + k + 2, i, j ) );
                  _r[ 3 ] = reduction( _r[ 3 ], dataFetcher( offset + k + 3, i, j ) );
               }
            }
         }
      }

      // reduction of the last, incomplete block (not unrolled)

      for( int i = 0; i < m; i++ ) {
         for( int j = 0; j < n; j++ ) {
            Result* _r = r.get() + ( i * n + j ) * 4;
            for( Index k = blocks * block_size; k < size; k++ )
               _r[ 0 ] = reduction( _r[ 0 ], dataFetcher( k, i, j ) );
         }
      }

      // reduction of unrolled results
      for( int i = 0; i < m; i++ ) {
         for( int j = 0; j < n; j++ ) {
            Result* _r = r.get() + ( i * n + j ) * 4;
            _r[ 0 ] = reduction( _r[ 0 ], _r[ 1 ] );
            _r[ 0 ] = reduction( _r[ 0 ], _r[ 2 ] );
            _r[ 0 ] = reduction( _r[ 0 ], _r[ 3 ] );

            // copy the result into the output parameter
            result( i, j ) = _r[ 0 ];
         }
      }
   }
   else {
      for( int i = 0; i < m; i++ ) {
         for( int j = 0; j < n; j++ ) {
            result( i, j ) = identity;
         }
      }

      for( int b = 0; b < blocks; b++ ) {
         const Index offset = b * block_size;
         for( int i = 0; i < m; i++ ) {
            for( int j = 0; j < n; j++ ) {
               for( int k = 0; k < block_size; k++ ) {
                  result( i, j ) = reduction( result( i, j ), dataFetcher( offset + k, i, j ) );
               }
            }
         }
      }

      for( int i = 0; i < m; i++ ) {
         for( int j = 0; j < n; j++ ) {
            for( int k = blocks * block_size; k < size; k++ ) {
               result( i, j ) = reduction( result( i, j ), dataFetcher( k, i, j ) );
            }
         }
      }
   }
#endif
}

template< typename Result, typename DataFetcher, typename Reduction, typename Index, typename Output >
void
Reduction3D< Devices::Host >::reduce( Result identity,
                                      DataFetcher dataFetcher,
                                      Reduction reduction,
                                      Index size,
                                      int m,
                                      int n,
                                      Output result )
{
   if( size < 0 )
      throw std::invalid_argument( "Reduction3D: The size of datasets must be non-negative." );
   if( n < 0 )
      throw std::invalid_argument( "Reduction3D: The number of datasets must be non-negative." );
   if( m < 0 )
      throw std::invalid_argument( "Reduction3D: The number of datasets must be non-negative." );

#ifdef HAVE_OPENMP
   constexpr int block_size = 128;
   const int blocks = size / block_size;

   if( Devices::Host::isOMPEnabled() && blocks >= 2 ) {
      const int threads = TNL::min( blocks, Devices::Host::getMaxThreadsCount() );
      #pragma omp parallel num_threads( threads )
      {
         // first thread initializes the result array
         #pragma omp single nowait
         {
            for( int i = 0; i < m; i++ ) {
               for( int j = 0; j < n; j++ ) {
                  result( i, j ) = identity;
               }
            }
         }

         // initialize array for thread-local results
         // (it is accessed as a row-major matrix with n rows and 4 columns)
         std::unique_ptr< Result[] > r{ new Result[ m * n * 4 ] };
         for( int i = 0; i < m * n * 4; i++ )
            r[ i ] = identity;

         #pragma omp for nowait
         for( int b = 0; b < blocks; b++ ) {
            const Index offset = b * block_size;
            for( int i = 0; i < m; i++ ) {
               for( int j = 0; j < n; j++ ) {
                  Result* _r = r.get() + ( i * n + j ) * 4;
                  for( int k = 0; k < block_size; k += 4 ) {
                     _r[ 0 ] = reduction( _r[ 0 ], dataFetcher( offset + k, i, j ) );
                     _r[ 1 ] = reduction( _r[ 1 ], dataFetcher( offset + k + 1, i, j ) );
                     _r[ 2 ] = reduction( _r[ 2 ], dataFetcher( offset + k + 2, i, j ) );
                     _r[ 3 ] = reduction( _r[ 3 ], dataFetcher( offset + k + 3, i, j ) );
                  }
               }
            }
         }

         // the first thread that reaches here processes the last, incomplete block
         #pragma omp single nowait
         {
            for( int i = 0; i < m; i++ ) {
               for( int j = 0; j < n; j++ ) {
                  Result* _r = r.get() + ( i * n + j ) * 4;
                  for( Index k = blocks * block_size; k < size; k++ )
                     _r[ 0 ] = reduction( _r[ 0 ], dataFetcher( k, i, j ) );
               }
            }
         }

         // local reduction of unrolled results
         for( int i = 0; i < m; i++ ) {
            for( int j = 0; j < n; j++ ) {
               Result* _r = r.get() + ( i * n + j ) * 4;
               _r[ 0 ] = reduction( _r[ 0 ], _r[ 1 ] );
               _r[ 0 ] = reduction( _r[ 0 ], _r[ 2 ] );
               _r[ 0 ] = reduction( _r[ 0 ], _r[ 3 ] );
            }
         }

         // inter-thread reduction of local results
         #pragma omp critical
         {
            for( int i = 0; i < m; i++ ) {
               for( int j = 0; j < n; j++ ) {
                  result( i, j ) = reduction( result( i, j ), r[ ( i * n + j ) * 4 ] );
               }
            }
         }
      }
   }
   else
#endif
      Reduction3D< Devices::Sequential >::reduce( identity, dataFetcher, reduction, size, m, n, result );
}

template< typename Result, typename DataFetcher, typename Reduction, typename Index, typename Output >
void
Reduction3D< Devices::Cuda >::reduce( Result identity,
                                      DataFetcher dataFetcher,
                                      Reduction reduction,
                                      Index size,
                                      int m,
                                      int n,
                                      Output hostResult )
{
   if( size < 0 )
      throw std::invalid_argument( "Reduction3D: The size of datasets must be non-negative." );
   if( n < 0 )
      throw std::invalid_argument( "Reduction3D: The number of datasets must be non-negative." );
   if( m < 0 )
      throw std::invalid_argument( "Reduction3D: The number of datasets must be non-negative." );

#ifdef CUDA_REDUCTION_PROFILING
   Timer timer;
   timer.reset();
   timer.start();
#endif

   Result* deviceAux1 = nullptr;
   const dim3 reducedSize = detail::CudaReduction3DKernelLauncher( identity, dataFetcher, reduction, size, m, n, deviceAux1 );

#ifdef CUDA_REDUCTION_PROFILING
   timer.stop();
   std::cout << "   Reduction3D of " << m << " and " << n << " datasets on GPU to size " << reducedSize << " took "
             << timer.getRealTime() << " sec. " << std::endl;
   timer.reset();
   timer.start();
#endif

   // transfer the reduced data from device to host
   std::unique_ptr< Result[] > resultArray{ new Result[ m * n * reducedSize.x ] };
   copy< void, Devices::Cuda >( resultArray.get(), deviceAux1, m * n * reducedSize.x );

#ifdef CUDA_REDUCTION_PROFILING
   timer.stop();
   std::cout << "   Transferring data to CPU took " << timer.getRealTime() << " sec. " << std::endl;
   timer.reset();
   timer.start();
#endif

   // finish the reduction on the host
   auto dataFetcherFinish = [ & ]( int x, int i, int j )
   {
      return resultArray[ x + j * reducedSize.x + i * reducedSize.x * n ];
   };
   Reduction3D< Devices::Sequential >::reduce(
      identity, dataFetcherFinish, reduction, Index( reducedSize.x ), m, n, hostResult );

#ifdef CUDA_REDUCTION_PROFILING
   timer.stop();
   std::cout << "   Reduction3D of small data set on CPU took " << timer.getRealTime() << " sec. " << std::endl;
#endif
}

}  // namespace TNL::Algorithms
