// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Algorithms {

/**
 * \brief Parallel for loop in n-dimensions.
 *
 * \tparam Device specifies the device where the for-loop will be executed.
 *    It can be \ref TNL::Devices::Host, \ref TNL::Devices::Cuda or
 *    \ref TNL::Devices::Sequential.
 */
template< typename Device = Devices::Sequential, bool expand = false >
struct ParallelForND
{
   /**
    * \brief Static method for the execution of the loop.
    *
    * \tparam Index is the type of the loop indices.
    * \tparam Function is the type of the functor to be called in each iteration
    *    (it is usually deduced from the argument used in the function call).
    * \tparam FunctionArgs is a variadic pack of types for additional parameters
    *    that are forwarded to the functor in every iteration.
    *
    * \param begin is the left bound of the iteration range `[begin, end)`.
    * \param end is the right bound of the iteration range `[begin, end)`.
    * \param f is the function to be called in each iteration.
    * \param args are additional parameters to be passed to the function f.
    *
    * \par Example
    * \include Algorithms/ParallelForExample.cpp
    * \par Output
    * \include ParallelForExample.out
    *
    */
   template< typename Coordinates, typename Function, typename... FunctionArgs >
   static void
   exec( const Coordinates& begin, const Coordinates& end, Function f, FunctionArgs... args )
   {
      constexpr int Dimension = Coordinates::getSize();
      Coordinates i;
      if constexpr( Dimension == 1 ) {
         for( i[ 0 ] = begin[ 0 ]; i[ 0 ] < end[ 0 ]; i[ 0 ]++ )
            if constexpr( expand )
               f( i[ 0 ], args... );
            else
               f( i, args... );
      }
      if constexpr( Dimension == 2 ) {
         for( i[ 1 ] = begin[ 1 ]; i[ 1 ] < end[ 1 ]; i[ 1 ]++ )
            for( i[ 0 ] = begin[ 0 ]; i[ 0 ] < end[ 0 ]; i[ 0 ]++ )
               if constexpr( expand )
                  f( i[ 0 ], i[ 1 ], args... );
               else
                  f( i, args... );
      }
      if constexpr( Dimension == 3 ) {
         for( i[ 2 ] = begin[ 2 ]; i[ 2 ] < end[ 2 ]; i[ 2 ]++ )
            for( i[ 1 ] = begin[ 1 ]; i[ 1 ] < end[ 1 ]; i[ 1 ]++ )
               for( i[ 0 ] = begin[ 0 ]; i[ 0 ] < end[ 0 ]; i[ 0 ]++ )
                  if constexpr( expand )
                     f( i[ 0 ], i[ 1 ], i[ 2 ], args... );
                  else
                     f( i, args... );
      }
      if constexpr( Dimension > 3 ) {
         i = begin;
         while( i[ Dimension - 1 ] < end[ Dimension - 1 ] ) {
            for( i[ 2 ] = begin[ 2 ]; i[ 2 ] < end[ 2 ]; i[ 2 ]++ )
               for( i[ 1 ] = begin[ 1 ]; i[ 1 ] < end[ 1 ]; i[ 1 ]++ )
                  for( i[ 0 ] = begin[ 0 ]; i[ 0 ] < end[ 0 ]; i[ 0 ]++ )
                     f( i, args... );  // TODO: implement expanded variant
            int idx = 3;
            i[ idx ]++;
            while( i[ idx ] == end[ idx ] && idx < Dimension - 1 ) {
               i[ idx ] = begin[ idx ];
               i[ ++idx ]++;
            }
         }
      }
   }

   /**
    * \brief Overload with custom launch configuration (which is ignored for
    * \ref TNL::Devices::Sequential).
    */
   template< typename Coordinates, typename Function, typename... FunctionArgs >
   static void
   exec( const Coordinates& begin,
         const Coordinates& end,
         typename Device::LaunchConfiguration launch_config,
         Function f,
         FunctionArgs... args )
   {
      exec( begin, end, f, args... );
   }
};

template< bool expand >
struct ParallelForND< Devices::Host, expand >
{
   template< typename Coordinates, typename Function, typename... FunctionArgs >
   static void
   exec( const Coordinates& begin,
         const Coordinates& end,
         Devices::Host::LaunchConfiguration launch_config,
         Function f,
         FunctionArgs... args )
   {
#ifdef HAVE_OPENMP
      using Index = typename Coordinates::IndexType;
      constexpr int Dimension = Coordinates::getSize();
      if constexpr( Dimension == 1 ) {
         // Benchmarks show that this is significantly faster compared
         // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() && end - start > 512 )'
         if( Devices::Host::isOMPEnabled() && end[ 0 ] - begin[ 0 ] > 512 ) {
   #pragma omp parallel for
            for( Index i = begin[ 0 ]; i < end[ 0 ]; i++ )
               if constexpr( expand )
                  f( i, args... );
               else {
                  Coordinates coordinates{ i };  // TODO: Move this outside the loop like in sequential version
                  f( coordinates, args... );
               }
         }
         else
            ParallelForND< Devices::Sequential, expand >::exec( begin, end, f, args... );
      }
      if constexpr( Dimension == 2 ) {
         // Benchmarks show that this is significantly faster compared
         // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() )'
         if( Devices::Host::isOMPEnabled() ) {
   #pragma omp parallel for
            for( Index j = begin[ 1 ]; j < end[ 1 ]; j++ ) {
               Coordinates c{ 0, j };  // TODO: Move this outside the loop like in sequential version
               for( c[ 0 ] = begin[ 0 ]; c[ 0 ] < end[ 0 ]; c[ 0 ]++ )
                  if constexpr( expand )
                     f( c[ 0 ], c[ 1 ], args... );
                  else
                     f( c, args... );
            }
         }
      }
      if constexpr( Dimension == 3 ) {
         // Benchmarks show that this is significantly faster compared
         // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() )'
         if( Devices::Host::isOMPEnabled() ) {
   #pragma omp parallel for
            for( Index k = begin[ 2 ]; k < end[ 2 ]; k++ ) {
               Coordinates c{ 0, 0, k };  // TODO: Move this outside the loop like in sequential version
               for( c[ 1 ] = begin[ 1 ]; c[ 1 ] < end[ 1 ]; c[ 1 ]++ )
                  for( c[ 0 ] = begin[ 0 ]; c[ 0 ] < end[ 0 ]; c[ 0 ]++ )
                     if constexpr( expand )
                        f( c[ 0 ], c[ 1 ], c[ 2 ], args... );
                     else
                        f( c, args... );
            }
         }
      }
      if constexpr( Dimension > 3 ) {
         // Benchmarks show that this is significantly faster compared
         // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() )'
         if( Devices::Host::isOMPEnabled() ) {
            Coordinates c = begin;
            while( c[ Dimension - 1 ] < end[ Dimension - 1 ] ) {
   #pragma omp parallel for firstprivate( c )
               for( Index k = begin[ 2 ]; k < end[ 2 ]; k++ ) {
                  Coordinates c1( c );
                  c1[ 2 ] = k;
                  for( c1[ 1 ] = begin[ 1 ]; c1[ 1 ] < end[ 1 ]; c1[ 1 ]++ )
                     for( c1[ 0 ] = begin[ 0 ]; c1[ 0 ] < end[ 0 ]; c1[ 0 ]++ ) {
                        f( c1, args... );  // TODO: implement expanded variant
                     }
               }
               int idx = 3;
               c[ idx ]++;
               while( c[ idx ] == end[ idx ] && idx < Dimension - 1 ) {
                  c[ idx ] = begin[ idx ];
                  c[ ++idx ]++;
               }
            }
         }
      }
#else
      ParallelForND< Devices::Sequential, expand >::exec( begin, end, f, args... );
#endif
   }

   template< typename Coordinates, typename Function, typename... FunctionArgs >
   static void
   exec( const Coordinates& begin, const Coordinates& end, Function f, FunctionArgs... args )
   {
      Devices::Host::LaunchConfiguration launch_config;
      exec( begin, end, launch_config, f, args... );
   }
};

template< bool gridStrideX = true,
          bool gridStrideY = true,
          bool gridStrideZ = true,
          bool expand,
          typename Coordinates,
          typename Function,
          typename... FunctionArgs >
__global__
void
ParallelForNDKernel( const Coordinates begin, const Coordinates end, Function f, FunctionArgs... args )
{
#ifdef __CUDACC__
   constexpr int Dimension = Coordinates::getSize();
   Coordinates i( begin );
   if constexpr( Dimension == 1 ) {
      i[ 0 ] = begin[ 0 ] + blockIdx.x * blockDim.x + threadIdx.x;
      while( i[ 0 ] < end[ 0 ] ) {
         if constexpr( expand )
            f( i[ 0 ], args... );
         else
            f( i, args... );
         if( gridStrideX )
            i[ 0 ] += blockDim.x * gridDim.x;
         else
            break;
      }
   }
   if constexpr( Dimension == 2 ) {
      i[ 1 ] = begin[ 1 ] + blockIdx.y * blockDim.y + threadIdx.y;
      i[ 0 ] = begin[ 0 ] + blockIdx.x * blockDim.x + threadIdx.x;
      while( i[ 1 ] < end[ 1 ] ) {
         while( i[ 0 ] < end[ 0 ] ) {
            if constexpr( expand )
               f( i[ 0 ], i[ 1 ], args... );
            else
               f( i, args... );
            if( gridStrideX )
               i[ 0 ] += blockDim.x * gridDim.x;
            else
               break;
         }
         if( gridStrideY )
            i[ 1 ] += blockDim.y * gridDim.y;
         else
            break;
      }
   }
   if constexpr( Dimension == 3 ) {
      i[ 2 ] = begin[ 2 ] + blockIdx.z * blockDim.z + threadIdx.z;
      i[ 1 ] = begin[ 1 ] + blockIdx.y * blockDim.y + threadIdx.y;
      i[ 0 ] = begin[ 0 ] + blockIdx.x * blockDim.x + threadIdx.x;
      while( i[ 2 ] < end[ 2 ] ) {
         while( i[ 1 ] < end[ 1 ] ) {
            while( i[ 0 ] < end[ 0 ] ) {
               if constexpr( expand )
                  f( i[ 0 ], i[ 1 ], i[ 2 ], args... );
               else
                  f( i, args... );
               if( gridStrideX )
                  i[ 0 ] += blockDim.x * gridDim.x;
               else
                  break;
            }
            if( gridStrideY )
               i[ 1 ] += blockDim.y * gridDim.y;
            else
               break;
         }
         if( gridStrideZ )
            i[ 2 ] += blockDim.z * gridDim.z;
         else
            break;
      }
   }
   if constexpr( Dimension > 3 ) {
      while( i[ Dimension - 1 ] < end[ Dimension - 1 ] ) {
         i[ 2 ] = begin[ 2 ] + blockIdx.z * blockDim.z + threadIdx.z;
         i[ 1 ] = begin[ 1 ] + blockIdx.y * blockDim.y + threadIdx.y;
         i[ 0 ] = begin[ 0 ] + blockIdx.x * blockDim.x + threadIdx.x;
         while( i[ 2 ] < end[ 2 ] ) {
            while( i[ 1 ] < end[ 1 ] ) {
               while( i[ 0 ] < end[ 0 ] ) {
                  f( i, args... );
                  if( gridStrideX )
                     i[ 0 ] += blockDim.x * gridDim.x;
                  else
                     break;
               }
               if( gridStrideY )
                  i[ 1 ] += blockDim.y * gridDim.y;
               else
                  break;
            }
            if( gridStrideZ )
               i[ 2 ] += blockDim.z * gridDim.z;
            else
               break;
         }
         int idx = 3;
         i[ idx ]++;
         while( i[ idx ] == end[ idx ] && idx < Dimension - 1 ) {
            i[ idx ] = begin[ idx ];
            i[ ++idx ]++;
         }
      }
   }
#endif
}

template< bool expand >
struct ParallelForND< Devices::Cuda, expand >
{
   // NOTE: launch_config must be passed by value so that the modifications of
   // blockSize and gridSize do not propagate to the caller
   template< typename Coordinates, typename Function, typename... FunctionArgs >
   static void
   exec( const Coordinates& begin,
         const Coordinates& end,
         Devices::Cuda::LaunchConfiguration launch_config,
         Function f,
         FunctionArgs... args )
   {
      using Index = typename Coordinates::IndexType;
      constexpr int Dimension = Coordinates::getSize();
      if constexpr( Dimension == 1 ) {
         if( end.x() <= begin.x() )
            return;

         launch_config.blockSize.x = 256;
         launch_config.blockSize.y = 1;
         launch_config.blockSize.z = 1;
         launch_config.gridSize.x = TNL::min( Backend::getMaxGridXSize(),
                                              Backend::getNumberOfBlocks( end[ 0 ] - begin[ 0 ], launch_config.blockSize.x ) );
         launch_config.gridSize.y = 1;
         launch_config.gridSize.z = 1;

         if( (std::size_t) launch_config.blockSize.x * launch_config.gridSize.x >= (std::size_t) end[ 0 ] - begin[ 0 ] ) {
            constexpr auto kernel = ParallelForNDKernel< false, false, false, expand, Coordinates, Function, FunctionArgs... >;
            Backend::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else {
            // decrease the grid size and align to the number of multiprocessors
            const int desGridSize = 32 * Backend::getDeviceMultiprocessors( Backend::getDevice() );
            launch_config.gridSize.x =
               TNL::min( desGridSize, Backend::getNumberOfBlocks( end[ 0 ] - begin[ 0 ], launch_config.blockSize.x ) );
            constexpr auto kernel = ParallelForNDKernel< true, false, false, expand, Coordinates, Function, FunctionArgs... >;
            Backend::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
      }
      if constexpr( Dimension == 2 ) {
         if( end.x() <= begin.x() || end.y() <= begin.y() )
            return;

         const Index sizeX = end[ 0 ] - begin[ 0 ];
         const Index sizeY = end[ 1 ] - begin[ 1 ];

         if( sizeX >= sizeY * sizeY ) {
            launch_config.blockSize.x = TNL::min( 256, sizeX );
            launch_config.blockSize.y = 1;
         }
         else if( sizeY >= sizeX * sizeX ) {
            launch_config.blockSize.x = 1;
            launch_config.blockSize.y = TNL::min( 256, sizeY );
         }
         else {
            launch_config.blockSize.x = TNL::min( 32, sizeX );
            launch_config.blockSize.y = TNL::min( 8, sizeY );
         }
         launch_config.blockSize.z = 1;
         launch_config.gridSize.x =
            TNL::min( Backend::getMaxGridXSize(), Backend::getNumberOfBlocks( sizeX, launch_config.blockSize.x ) );
         launch_config.gridSize.y =
            TNL::min( Backend::getMaxGridYSize(), Backend::getNumberOfBlocks( sizeY, launch_config.blockSize.y ) );
         launch_config.gridSize.z = 1;

         dim3 gridCount;
         gridCount.x = roundUpDivision( sizeX, launch_config.blockSize.x * launch_config.gridSize.x );
         gridCount.y = roundUpDivision( sizeY, launch_config.blockSize.y * launch_config.gridSize.y );

         if( gridCount.x == 1 && gridCount.y == 1 ) {
            constexpr auto kernel = ParallelForNDKernel< false, false, false, expand, Coordinates, Function, FunctionArgs... >;
            Backend::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x == 1 && gridCount.y > 1 ) {
            constexpr auto kernel = ParallelForNDKernel< false, true, false, expand, Coordinates, Function, FunctionArgs... >;
            Backend::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x > 1 && gridCount.y == 1 ) {
            constexpr auto kernel = ParallelForNDKernel< true, false, false, expand, Coordinates, Function, FunctionArgs... >;
            Backend::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else {
            constexpr auto kernel = ParallelForNDKernel< true, true, false, expand, Coordinates, Function, FunctionArgs... >;
            Backend::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
      }
      if constexpr( Dimension >= 3 ) {
         for( int i = 0; i < Dimension; i++ )
            if( end[ i ] <= begin[ i ] )
               return;

         const Index sizeX = end[ 0 ] - begin[ 0 ];
         const Index sizeY = end[ 1 ] - begin[ 1 ];
         const Index sizeZ = end[ 2 ] - begin[ 2 ];

         if( sizeX >= sizeY * sizeY * sizeZ * sizeZ ) {
            launch_config.blockSize.x = TNL::min( 256, sizeX );
            launch_config.blockSize.y = 1;
            launch_config.blockSize.z = 1;
         }
         else if( sizeY >= sizeX * sizeX * sizeZ * sizeZ ) {
            launch_config.blockSize.x = 1;
            launch_config.blockSize.y = TNL::min( 256, sizeY );
            launch_config.blockSize.z = 1;
         }
         else if( sizeZ >= sizeX * sizeX * sizeY * sizeY ) {
            launch_config.blockSize.x = TNL::min( 2, sizeX );
            launch_config.blockSize.y = TNL::min( 2, sizeY );
            // CUDA allows max 64 for launch_config.blockSize.z
            launch_config.blockSize.z = TNL::min( 64, sizeZ );
         }
         else if( sizeX >= sizeZ * sizeZ && sizeY >= sizeZ * sizeZ ) {
            launch_config.blockSize.x = TNL::min( 32, sizeX );
            launch_config.blockSize.y = TNL::min( 8, sizeY );
            launch_config.blockSize.z = 1;
         }
         else if( sizeX >= sizeY * sizeY && sizeZ >= sizeY * sizeY ) {
            launch_config.blockSize.x = TNL::min( 32, sizeX );
            launch_config.blockSize.y = 1;
            launch_config.blockSize.z = TNL::min( 8, sizeZ );
         }
         else if( sizeY >= sizeX * sizeX && sizeZ >= sizeX * sizeX ) {
            launch_config.blockSize.x = 1;
            launch_config.blockSize.y = TNL::min( 32, sizeY );
            launch_config.blockSize.z = TNL::min( 8, sizeZ );
         }
         else {
            launch_config.blockSize.x = TNL::min( 16, sizeX );
            launch_config.blockSize.y = TNL::min( 4, sizeY );
            launch_config.blockSize.z = TNL::min( 4, sizeZ );
         }
         launch_config.gridSize.x =
            TNL::min( Backend::getMaxGridXSize(), Backend::getNumberOfBlocks( sizeX, launch_config.blockSize.x ) );
         launch_config.gridSize.y =
            TNL::min( Backend::getMaxGridYSize(), Backend::getNumberOfBlocks( sizeY, launch_config.blockSize.y ) );
         launch_config.gridSize.z =
            TNL::min( Backend::getMaxGridZSize(), Backend::getNumberOfBlocks( sizeZ, launch_config.blockSize.z ) );

         dim3 gridCount;
         gridCount.x = roundUpDivision( sizeX, launch_config.blockSize.x * launch_config.gridSize.x );
         gridCount.y = roundUpDivision( sizeY, launch_config.blockSize.y * launch_config.gridSize.y );
         gridCount.z = roundUpDivision( sizeZ, launch_config.blockSize.z * launch_config.gridSize.z );

         if( gridCount.x == 1 && gridCount.y == 1 && gridCount.z == 1 ) {
            constexpr auto kernel = ParallelForNDKernel< false, false, false, expand, Coordinates, Function, FunctionArgs... >;
            Backend::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x == 1 && gridCount.y == 1 && gridCount.z > 1 ) {
            constexpr auto kernel = ParallelForNDKernel< false, false, true, expand, Coordinates, Function, FunctionArgs... >;
            Backend::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x == 1 && gridCount.y > 1 && gridCount.z == 1 ) {
            constexpr auto kernel = ParallelForNDKernel< false, true, false, expand, Coordinates, Function, FunctionArgs... >;
            Backend::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x > 1 && gridCount.y == 1 && gridCount.z == 1 ) {
            constexpr auto kernel = ParallelForNDKernel< true, false, false, expand, Coordinates, Function, FunctionArgs... >;
            Backend::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x == 1 && gridCount.y > 1 && gridCount.z > 1 ) {
            constexpr auto kernel = ParallelForNDKernel< false, true, true, expand, Coordinates, Function, FunctionArgs... >;
            Backend::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x > 1 && gridCount.y > 1 && gridCount.z == 1 ) {
            constexpr auto kernel = ParallelForNDKernel< true, true, false, expand, Coordinates, Function, FunctionArgs... >;
            Backend::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x > 1 && gridCount.y == 1 && gridCount.z > 1 ) {
            constexpr auto kernel = ParallelForNDKernel< true, false, true, expand, Coordinates, Function, FunctionArgs... >;
            Backend::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else {
            constexpr auto kernel = ParallelForNDKernel< true, true, true, expand, Coordinates, Function, FunctionArgs... >;
            Backend::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
      }
   }

   template< typename Coordinates, typename Function, typename... FunctionArgs >
   static void
   exec( const Coordinates& begin, const Coordinates& end, Function f, FunctionArgs... args )
   {
      Devices::Cuda::LaunchConfiguration launch_config;
      exec( begin, end, launch_config, f, args... );
   }
};
}  //namespace Algorithms
}  // namespace TNL
