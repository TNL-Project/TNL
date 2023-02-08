// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Cuda/DeviceInfo.h>
#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Cuda/KernelLaunch.h>
#include <TNL/Math.h>

namespace TNL {
namespace Algorithms {
namespace detail {

template< typename Device = Devices::Sequential >
struct ParallelFor2D
{
   template< typename Index, typename Function, typename... FunctionArgs >
   static void
   exec( Index startX,
         Index startY,
         Index endX,
         Index endY,
         typename Device::LaunchConfiguration launch_config,
         Function f,
         FunctionArgs... args )
   {
      for( Index j = startY; j < endY; j++ )
         for( Index i = startX; i < endX; i++ )
            f( i, j, args... );
   }
};

template<>
struct ParallelFor2D< Devices::Host >
{
   template< typename Index, typename Function, typename... FunctionArgs >
   static void
   exec( Index startX,
         Index startY,
         Index endX,
         Index endY,
         Devices::Host::LaunchConfiguration launch_config,
         Function f,
         FunctionArgs... args )
   {
#ifdef HAVE_OPENMP
      // Benchmarks show that this is significantly faster compared
      // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() )'
      if( Devices::Host::isOMPEnabled() ) {
         #pragma omp parallel for
         for( Index j = startY; j < endY; j++ )
            for( Index i = startX; i < endX; i++ )
               f( i, j, args... );
      }
      else {
         Devices::Sequential::LaunchConfiguration sequential_config;
         ParallelFor2D< Devices::Sequential >::exec( startX, startY, endX, endY, sequential_config, f, args... );
      }
#else
      Devices::Sequential::LaunchConfiguration sequential_config;
      ParallelFor2D< Devices::Sequential >::exec( startX, startY, endX, endY, sequential_config, f, args... );
#endif
   }
};

template< bool gridStrideX = true, bool gridStrideY = true, typename Index, typename Function, typename... FunctionArgs >
__global__
void
ParallelFor2DKernel( Index startX, Index startY, Index endX, Index endY, Function f, FunctionArgs... args )
{
#ifdef __CUDACC__
   Index j = startY + blockIdx.y * blockDim.y + threadIdx.y;
   Index i = startX + blockIdx.x * blockDim.x + threadIdx.x;
   while( j < endY ) {
      while( i < endX ) {
         f( i, j, args... );
         if( gridStrideX )
            i += blockDim.x * gridDim.x;
         else
            break;
      }
      if( gridStrideY )
         j += blockDim.y * gridDim.y;
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
   template< typename Index, typename Function, typename... FunctionArgs >
   static void
   exec( Index startX,
         Index startY,
         Index endX,
         Index endY,
         Devices::Cuda::LaunchConfiguration launch_config,
         Function f,
         FunctionArgs... args )
   {
      if( endX <= startX || endY <= startY )
         return;

      const Index sizeX = endX - startX;
      const Index sizeY = endY - startY;

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
         constexpr auto kernel = ParallelFor2DKernel< false, false, Index, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, startX, startY, endX, endY, f, args... );
      }
      else if( gridCount.x == 1 && gridCount.y > 1 ) {
         constexpr auto kernel = ParallelFor2DKernel< false, true, Index, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, startX, startY, endX, endY, f, args... );
      }
      else if( gridCount.x > 1 && gridCount.y == 1 ) {
         constexpr auto kernel = ParallelFor2DKernel< true, false, Index, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, startX, startY, endX, endY, f, args... );
      }
      else {
         constexpr auto kernel = ParallelFor2DKernel< true, true, Index, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, startX, startY, endX, endY, f, args... );
      }
   }
};

}  // namespace detail
}  // namespace Algorithms
}  // namespace TNL
