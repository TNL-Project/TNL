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
   exec( const Coordinates& begin, const Coordinates& end,
         Function f, FunctionArgs... args )
   {
      constexpr int Dimension = Coordinates::getSize();
      Coordinates i;
      if constexpr( Dimension == 1 )
      {
         for( i.x() = begin.x(); i.x() < end.x(); i.x()++ )
            if constexpr( expand )
               f( i.x(), args... );
            else
               f( i, args... );
      }
      if constexpr( Dimension == 2 )
      {
         for( i.y() = begin.y(); i.y() < end.y(); i.y()++ )
           for( i.x() = begin.x(); i.x() < end.x(); i.x()++ )
               if constexpr( expand )
                  f( i.x(), i.y(), args... );
               else
                  f( i, args... );
      }
      if constexpr( Dimension == 3 )
      {
         for( i.z() = begin.z(); i.z() < end.z(); i.z()++ )
            for( i.y() = begin.y(); i.y() < end.y(); i.y()++ )
               for( i.x() = begin.x(); i.x() < end.x(); i.x()++ )
                  if constexpr( expand )
                     f( i.x(), i.y(), i.z(), args... );
                  else
                     f( i, args... );
      }
      if constexpr( Dimension > 3 )
      {
         i = begin;
         while( i[ Dimension-1 ] < end[ Dimension-1 ] ) {
            for( i.z() = begin.z(); i.z() < end.z(); i.z()++ )
               for( i.y() = begin.y(); i.y() < end.y(); i.y()++ )
                  for( i.x() = begin.x(); i.x() < end.x(); i.x()++ )
                     f( i, args... ); // TODO: implement expanded variant
            int idx = 3;
            i[ idx ]++;
            while( i[ idx ] == end[ idx ] && idx < Dimension-1 ) {
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
   static void exec( const Coordinates& begin, const Coordinates& end,
                     typename Device::LaunchConfiguration launch_config, Function f, FunctionArgs... args )
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
         Function f, FunctionArgs... args )
   {
#ifdef HAVE_OPENMP
      using Index = typename Coordinates::IndexType;
      constexpr int Dimension = Coordinates::getSize();
      if constexpr( Dimension == 1 )
      {
         // Benchmarks show that this is significantly faster compared
         // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() && end - start > 512 )'
         if( Devices::Host::isOMPEnabled() && end.x() - begin.x() > 512 ) {
            #pragma omp parallel for
            for( Index i = begin.x(); i < end.x(); i++ )
               if constexpr( expand )
                  f( i, args... );
               else {
                  Coordinates coordinates{ i }; // TODO: Move this outside the loop like in sequential version
                  f( coordinates, args... );
               }
         }
         else
            ParallelForND< Devices::Sequential, expand >::exec( begin, end, f, args... );
      }
      if constexpr( Dimension == 2 )
      {
         // Benchmarks show that this is significantly faster compared
         // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() )'
         if( Devices::Host::isOMPEnabled() ) {
            #pragma omp parallel for
            for( Index j = begin.y(); j < end.y(); j++ )
               for( Index i = begin.x(); i < end.x(); i++ )
                  if constexpr( expand )
                     f( i, j, args... );
                  else {
                     Coordinates coordinates{ i, j };  // TODO: Move this outside the loop like in sequential version
                     f( coordinates, args... );
                  }
         }
      }
      if constexpr( Dimension == 3 )
      {
         // Benchmarks show that this is significantly faster compared
         // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() )'
         if( Devices::Host::isOMPEnabled() ) {
            #pragma omp parallel for
            for( Index k = begin.z(); k < end.z(); k++ )
               for( Index j = begin.y(); j < end.y(); j++ )
                  for( Index i = begin.x(); i < end.x(); i++ )
                     if constexpr( expand )
                        f( i, j, k, args... );
                     else {
                        Coordinates coordinates{ i, j, k };  // TODO: Move this outside the loop like in sequential version
                        f( coordinates, args... );
                     }
         }
      }
      if constexpr( Dimension > 3 )
      {
         // Benchmarks show that this is significantly faster compared
         // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() )'
         if( Devices::Host::isOMPEnabled() ) {
            #pragma omp parallel for
            for( Index k = begin[ Dimension-1 ]; k < end[ Dimension-1 ]; k++ )
               for( Index j = begin[ Dimension-2 ]; j < end[ Dimension-2 ]; j++ )
                  for( Index i = begin[ Dimension-3 ]; i < end[ Dimension-3 ]; i++ )
                  {
                     Coordinates c( begin );
                     c[ Dimension-1 ] = k;
                     c[ Dimension-2 ] = j;
                     c[ Dimension-3 ] = i;
                     while( c[ Dimension-4] < end[ Dimension-4 ] ) {
                        f( c, args... );
                        int idx = 0;
                        c[ idx ]++;
                        while( c[ idx ] == end[ idx ] && idx < Dimension-4 ) {
                           c[ idx ] = begin[ idx ];
                           c[ ++idx ]++;
                        }
                     }
                  } // TODO: implement expanded variant
         }
      }
#else
      ParallelForND< Devices::Sequential, expand >::exec( begin, end, f, args... );
#endif
   }

   template< typename Coordinates, typename Function, typename... FunctionArgs >
   static void
   exec( const Coordinates& begin, const Coordinates& end,
         Function f, FunctionArgs... args )
   {
      Devices::Host::LaunchConfiguration launch_config;
      exec( begin, end, launch_config, f, args... );
   }
};

template< bool gridStrideX = true,
          bool gridStrideY = true,
          bool gridStrideZ = true,
          bool expand, typename Coordinates, typename Function, typename... FunctionArgs >
__global__
void
ParallelForNDKernel( const Coordinates begin, const Coordinates end, Function f, FunctionArgs... args )
{
#ifdef HAVE_CUDA
   constexpr int Dimension = Coordinates::getSize();
   Coordinates i( begin );
   if constexpr( Dimension == 1 )
   {
      i.x() = begin.x() + blockIdx.x * blockDim.x + threadIdx.x;
      while( i.x() < end.x() ) {
         if constexpr( expand )
            f( i.x(), args... );
         else
            f( i, args... );
         if( gridStrideX )
            i.x() += blockDim.x * gridDim.x;
         else
            break;
      }
   }
   if constexpr( Dimension == 2 )
   {
      i.y() = begin.y() + blockIdx.y * blockDim.y + threadIdx.y;
      i.x() = begin.x() + blockIdx.x * blockDim.x + threadIdx.x;
      while( i.y() < end.y() ) {
         while( i.x() < end.x() ) {
            if constexpr( expand )
               f( i.x(), i.y(), args... );
            else
               f( i, args... );
            if( gridStrideX )
               i.x() += blockDim.x * gridDim.x;
            else
               break;
         }
         if( gridStrideY )
            i.y() += blockDim.y * gridDim.y;
         else
            break;
      }
   }
   if constexpr( Dimension == 3 )
   {
      i.z() = begin.z() + blockIdx.z * blockDim.z + threadIdx.z;
      i.y() = begin.y() + blockIdx.y * blockDim.y + threadIdx.y;
      i.x() = begin.x() + blockIdx.x * blockDim.x + threadIdx.x;
      while( i.z() < end.z() ) {
         while( i.y() < end.y() ) {
            while( i.x() < end.x() ) {
               if constexpr( expand )
                  f( i.x(), i.y(), i.z(), args... );
               else
                  f( i, args... );
               if( gridStrideX )
                  i.x() += blockDim.x * gridDim.x;
               else
                  break;
            }
            if( gridStrideY )
               i.y() += blockDim.y * gridDim.y;
            else
               break;
         }
         if( gridStrideZ )
            i.z() += blockDim.z * gridDim.z;
         else
            break;
      }
   }
   if constexpr( Dimension > 3 )
   {
      while( i[Dimension-1] < end[Dimension-1])
      {
         i.z() = begin.z() + blockIdx.z * blockDim.z + threadIdx.z;
         i.y() = begin.y() + blockIdx.y * blockDim.y + threadIdx.y;
         i.x() = begin.x() + blockIdx.x * blockDim.x + threadIdx.x;
         while( i.z() < end.z() ) {
            while( i.y() < end.y() ) {
               while( i.x() < end.x() ) {
                  f( i, args... );
                  if( gridStrideX )
                     i.x() += blockDim.x * gridDim.x;
                  else
                     break;
               }
               if( gridStrideY )
                  i.y() += blockDim.y * gridDim.y;
               else
                  break;
            }
            if( gridStrideZ )
               i.z() += blockDim.z * gridDim.z;
            else
               break;
         }
         int idx = 3;
         i[ idx ]++;
         while( i[ idx ] == end[ idx ] && idx < Dimension-1 ) {
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
   exec( const Coordinates& begin, const Coordinates& end,
         Devices::Cuda::LaunchConfiguration launch_config,
         Function f, FunctionArgs... args )
   {
      using Index = typename Coordinates::IndexType;
      constexpr int Dimension = Coordinates::getSize();
      if constexpr( Dimension == 1 ) {
         if( end.x() <= begin.x() )
            return;

         launch_config.blockSize.x = 256;
         launch_config.blockSize.y = 1;
         launch_config.blockSize.z = 1;
         launch_config.gridSize.x =
            TNL::min( Cuda::getMaxGridXSize(), Cuda::getNumberOfBlocks( end.x() - begin.x(), launch_config.blockSize.x ) );
         launch_config.gridSize.y = 1;
         launch_config.gridSize.z = 1;

         if( (std::size_t) launch_config.blockSize.x * launch_config.gridSize.x >= (std::size_t) end.x() - begin.x() ) {
            constexpr auto kernel = ParallelForNDKernel< false, false, false, expand, Coordinates, Function, FunctionArgs... >;
            Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else {
            // decrease the grid size and align to the number of multiprocessors
            const int desGridSize = 32 * Cuda::DeviceInfo::getCudaMultiprocessors( Cuda::DeviceInfo::getActiveDevice() );
            launch_config.gridSize.x = TNL::min( desGridSize, Cuda::getNumberOfBlocks( end.x() - begin.x(), launch_config.blockSize.x ) );
            constexpr auto kernel = ParallelForNDKernel< true, false, false, expand, Coordinates, Function, FunctionArgs... >;
            Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
      }
      if constexpr( Dimension == 2 ) {
         if( end.x() <= begin.x() || end.y() <= begin.y() )
         return;

         const Index sizeX = end.x() - begin.x();
         const Index sizeY = end.y() - begin.y();

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
            TNL::min( Cuda::getMaxGridXSize(), Cuda::getNumberOfBlocks( sizeX, launch_config.blockSize.x ) );
         launch_config.gridSize.y =
            TNL::min( Cuda::getMaxGridYSize(), Cuda::getNumberOfBlocks( sizeY, launch_config.blockSize.y ) );
         launch_config.gridSize.z = 1;

         dim3 gridCount;
         gridCount.x = roundUpDivision( sizeX, launch_config.blockSize.x * launch_config.gridSize.x );
         gridCount.y = roundUpDivision( sizeY, launch_config.blockSize.y * launch_config.gridSize.y );

         if( gridCount.x == 1 && gridCount.y == 1 ) {
            constexpr auto kernel = ParallelForNDKernel< false, false, false, expand, Coordinates, Function, FunctionArgs... >;
            Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x == 1 && gridCount.y > 1 ) {
            constexpr auto kernel = ParallelForNDKernel< false, true, false, expand, Coordinates, Function, FunctionArgs... >;
            Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x > 1 && gridCount.y == 1 ) {
            constexpr auto kernel = ParallelForNDKernel< true, false, false, expand, Coordinates, Function, FunctionArgs... >;
            Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else {
            constexpr auto kernel = ParallelForNDKernel< true, true, false, expand, Coordinates, Function, FunctionArgs... >;
            Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
      }
      if constexpr( Dimension >= 3 ) {
         if( ! ( end > begin ) )
            return;

         const Index sizeX = end.x() - begin.x();
         const Index sizeY = end.y() - begin.y();
         const Index sizeZ = end.z() - begin.z();

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
            TNL::min( Cuda::getMaxGridXSize(), Cuda::getNumberOfBlocks( sizeX, launch_config.blockSize.x ) );
         launch_config.gridSize.y =
            TNL::min( Cuda::getMaxGridYSize(), Cuda::getNumberOfBlocks( sizeY, launch_config.blockSize.y ) );
         launch_config.gridSize.z =
            TNL::min( Cuda::getMaxGridZSize(), Cuda::getNumberOfBlocks( sizeZ, launch_config.blockSize.z ) );

         dim3 gridCount;
         gridCount.x = roundUpDivision( sizeX, launch_config.blockSize.x * launch_config.gridSize.x );
         gridCount.y = roundUpDivision( sizeY, launch_config.blockSize.y * launch_config.gridSize.y );
         gridCount.z = roundUpDivision( sizeZ, launch_config.blockSize.z * launch_config.gridSize.z );

         if( gridCount.x == 1 && gridCount.y == 1 && gridCount.z == 1 ) {
            constexpr auto kernel = ParallelForNDKernel< false, false, false, expand, Coordinates, Function, FunctionArgs... >;
            Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x == 1 && gridCount.y == 1 && gridCount.z > 1 ) {
            constexpr auto kernel = ParallelForNDKernel< false, false, true, expand, Coordinates, Function, FunctionArgs... >;
            Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x == 1 && gridCount.y > 1 && gridCount.z == 1 ) {
            constexpr auto kernel = ParallelForNDKernel< false, true, false, expand, Coordinates, Function, FunctionArgs... >;
            Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x > 1 && gridCount.y == 1 && gridCount.z == 1 ) {
            constexpr auto kernel = ParallelForNDKernel< true, false, false, expand, Coordinates, Function, FunctionArgs... >;
            Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x == 1 && gridCount.y > 1 && gridCount.z > 1 ) {
            constexpr auto kernel = ParallelForNDKernel< false, true, true, expand, Coordinates, Function, FunctionArgs... >;
            Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x > 1 && gridCount.y > 1 && gridCount.z == 1 ) {
            constexpr auto kernel = ParallelForNDKernel< true, true, false, expand, Coordinates, Function, FunctionArgs... >;
            Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else if( gridCount.x > 1 && gridCount.y == 1 && gridCount.z > 1 ) {
            constexpr auto kernel = ParallelForNDKernel< true, false, true, expand, Coordinates, Function, FunctionArgs... >;
            Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
         else {
            constexpr auto kernel = ParallelForNDKernel< true, true, true, expand, Coordinates, Function, FunctionArgs... >;
            Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
         }
      }
   }

   template< typename Coordinates, typename Function, typename... FunctionArgs >
   static void
   exec( const Coordinates& begin, const Coordinates& end,
         Function f, FunctionArgs... args )
   {
      Devices::Cuda::LaunchConfiguration launch_config;
      exec( begin, end, launch_config, f, args... );
   }
};
   } //namespace Algorithms
} // namespace TNL
