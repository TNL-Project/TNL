// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Cuda/KernelLaunch.h>
#include <TNL/Math.h>

namespace TNL::Algorithms::detail {

template< typename Device = Devices::Sequential >
struct ParallelFor2D
{
   template< typename MultiIndex, typename Function, typename... FunctionArgs >
   static void
   exec( MultiIndex begin,
         MultiIndex end,
         typename Device::LaunchConfiguration launch_config,
         Function f,
         FunctionArgs... args )
   {
      static_assert( MultiIndex::getSize() == 2, "ParallelFor2D requires a multi-index of size 2" );

      MultiIndex i;
      for( i.y() = begin.y(); i.y() < end.y(); i.y()++ )
         for( i.x() = begin.x(); i.x() < end.x(); i.x()++ )
            f( i, args... );
   }
};

template<>
struct ParallelFor2D< Devices::Host >
{
   template< typename MultiIndex, typename Function, typename... FunctionArgs >
   static void
   exec( MultiIndex begin, MultiIndex end, Devices::Host::LaunchConfiguration launch_config, Function f, FunctionArgs... args )
   {
      static_assert( MultiIndex::getSize() == 2, "ParallelFor2D requires a multi-index of size 2" );

#ifdef HAVE_OPENMP
      // Benchmarks show that this is significantly faster compared
      // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() )'
      if( Devices::Host::isOMPEnabled() ) {
         using Index = typename MultiIndex::ValueType;
         #pragma omp parallel for
         for( Index y = begin.y(); y < end.y(); y++ ) {
            MultiIndex i{ begin.x(), y };
            for( ; i.x() < end.x(); i.x()++ )
               f( i, args... );
         }
      }
      else {
         Devices::Sequential::LaunchConfiguration sequential_config;
         ParallelFor2D< Devices::Sequential >::exec( begin, end, sequential_config, f, args... );
      }
#else
      Devices::Sequential::LaunchConfiguration sequential_config;
      ParallelFor2D< Devices::Sequential >::exec( begin, end, sequential_config, f, args... );
#endif
   }
};

template< bool gridStride = true, typename MultiIndex, typename Function, typename... FunctionArgs >
__global__
void
ParallelFor2DKernel( MultiIndex begin, MultiIndex end, Function f, FunctionArgs... args )
{
#ifdef __CUDACC__
   // shift begin to the thread's initial position
   begin.x() += blockIdx.x * blockDim.x + threadIdx.x;
   begin.y() += blockIdx.y * blockDim.y + threadIdx.y;

   // initialize iteration index
   MultiIndex i = begin;

   while( i.y() < end.y() ) {
      while( i.x() < end.x() ) {
         f( i, args... );
         if( gridStride )
            i.x() += blockDim.x * gridDim.x;
         else
            break;
      }
      if( gridStride ) {
         i.x() = begin.x();
         i.y() += blockDim.y * gridDim.y;
      }
      else
         break;
   }
#endif
}

template<>
struct ParallelFor2D< Devices::Cuda >
{
   // NOTE: launch_config must be passed by value so that the modifications of
   // blockSize and gridSize do not propagate to the caller
   template< typename MultiIndex, typename Function, typename... FunctionArgs >
   static void
   exec( MultiIndex begin, MultiIndex end, Devices::Cuda::LaunchConfiguration launch_config, Function f, FunctionArgs... args )
   {
      static_assert( MultiIndex::getSize() == 2, "ParallelFor2D requires a multi-index of size 2" );

      if( end.x() <= begin.x() || end.y() <= begin.y() )
         return;

      const auto sizeX = end.x() - begin.x();
      const auto sizeY = end.y() - begin.y();

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
         constexpr auto kernel = ParallelFor2DKernel< false, MultiIndex, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
      }
      else {
         constexpr auto kernel = ParallelFor2DKernel< true, MultiIndex, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
      }
   }
};

}  // namespace TNL::Algorithms::detail
