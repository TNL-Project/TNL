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
struct ParallelFor3D
{
   template< typename Index, typename Function, typename... FunctionArgs >
   static void
   exec( Index startX,
         Index startY,
         Index startZ,
         Index endX,
         Index endY,
         Index endZ,
         typename Device::LaunchConfiguration launch_config,
         Function f,
         FunctionArgs... args )
   {
      for( Index k = startZ; k < endZ; k++ )
         for( Index j = startY; j < endY; j++ )
            for( Index i = startX; i < endX; i++ )
               f( i, j, k, args... );
   }
};

template<>
struct ParallelFor3D< Devices::Host >
{
   template< typename Index, typename Function, typename... FunctionArgs >
   static void
   exec( Index startX,
         Index startY,
         Index startZ,
         Index endX,
         Index endY,
         Index endZ,
         Devices::Host::LaunchConfiguration launch_config,
         Function f,
         FunctionArgs... args )
   {
#ifdef HAVE_OPENMP
      // Benchmarks show that this is significantly faster compared
      // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() )'
      if( Devices::Host::isOMPEnabled() ) {
         #pragma omp parallel for collapse( 2 )
         for( Index k = startZ; k < endZ; k++ )
            for( Index j = startY; j < endY; j++ )
               for( Index i = startX; i < endX; i++ )
                  f( i, j, k, args... );
      }
      else {
         Devices::Sequential::LaunchConfiguration sequential_config;
         ParallelFor3D< Devices::Sequential >::exec( startX, startY, startZ, endX, endY, endZ, sequential_config, f, args... );
      }
#else
      Devices::Sequential::LaunchConfiguration sequential_config;
      ParallelFor3D< Devices::Sequential >::exec( startX, startY, startZ, endX, endY, endZ, sequential_config, f, args... );
#endif
   }
};

template< bool gridStrideX = true,
          bool gridStrideY = true,
          bool gridStrideZ = true,
          typename Index,
          typename Function,
          typename... FunctionArgs >
__global__
void
ParallelFor3DKernel( Index startX,
                     Index startY,
                     Index startZ,
                     Index endX,
                     Index endY,
                     Index endZ,
                     Function f,
                     FunctionArgs... args )
{
#ifdef __CUDACC__
   Index k = startZ + blockIdx.z * blockDim.z + threadIdx.z;
   Index j = startY + blockIdx.y * blockDim.y + threadIdx.y;
   Index i = startX + blockIdx.x * blockDim.x + threadIdx.x;
   while( k < endZ ) {
      while( j < endY ) {
         while( i < endX ) {
            f( i, j, k, args... );
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
      if( gridStrideZ )
         k += blockDim.z * gridDim.z;
      else
         break;
   }
#endif
}

template<>
struct ParallelFor3D< Devices::Cuda >
{
   // NOTE: launch_config must be passed by value so that the modifications of
   // blockSize and gridSize do not propagate to the caller
   template< typename Index, typename Function, typename... FunctionArgs >
   static void
   exec( Index startX,
         Index startY,
         Index startZ,
         Index endX,
         Index endY,
         Index endZ,
         Devices::Cuda::LaunchConfiguration launch_config,
         Function f,
         FunctionArgs... args )
   {
      if( endX <= startX || endY <= startY || endZ <= startZ )
         return;

      const Index sizeX = endX - startX;
      const Index sizeY = endY - startY;
      const Index sizeZ = endZ - startZ;

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
         constexpr auto kernel = ParallelFor3DKernel< false, false, false, Index, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, startX, startY, startZ, endX, endY, endZ, f, args... );
      }
      else if( gridCount.x == 1 && gridCount.y == 1 && gridCount.z > 1 ) {
         constexpr auto kernel = ParallelFor3DKernel< false, false, true, Index, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, startX, startY, startZ, endX, endY, endZ, f, args... );
      }
      else if( gridCount.x == 1 && gridCount.y > 1 && gridCount.z == 1 ) {
         constexpr auto kernel = ParallelFor3DKernel< false, true, false, Index, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, startX, startY, startZ, endX, endY, endZ, f, args... );
      }
      else if( gridCount.x > 1 && gridCount.y == 1 && gridCount.z == 1 ) {
         constexpr auto kernel = ParallelFor3DKernel< true, false, false, Index, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, startX, startY, startZ, endX, endY, endZ, f, args... );
      }
      else if( gridCount.x == 1 && gridCount.y > 1 && gridCount.z > 1 ) {
         constexpr auto kernel = ParallelFor3DKernel< false, true, true, Index, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, startX, startY, startZ, endX, endY, endZ, f, args... );
      }
      else if( gridCount.x > 1 && gridCount.y > 1 && gridCount.z == 1 ) {
         constexpr auto kernel = ParallelFor3DKernel< true, true, false, Index, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, startX, startY, startZ, endX, endY, endZ, f, args... );
      }
      else if( gridCount.x > 1 && gridCount.y == 1 && gridCount.z > 1 ) {
         constexpr auto kernel = ParallelFor3DKernel< true, false, true, Index, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, startX, startY, startZ, endX, endY, endZ, f, args... );
      }
      else {
         constexpr auto kernel = ParallelFor3DKernel< true, true, true, Index, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, startX, startY, startZ, endX, endY, endZ, f, args... );
      }
   }
};

}  // namespace detail
}  // namespace Algorithms
}  // namespace TNL
