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

namespace TNL::Algorithms::detail {

template< typename Device = Devices::Sequential >
struct ParallelFor3D
{
   template< typename MultiIndex, typename Function, typename... FunctionArgs >
   static void
   exec( MultiIndex begin,
         MultiIndex end,
         typename Device::LaunchConfiguration launch_config,
         Function f,
         FunctionArgs... args )
   {
      static_assert( MultiIndex::getSize() == 3, "ParallelFor3D requires a multi-index of size 3" );

      MultiIndex i;
      for( i.z() = begin.z(); i.z() < end.z(); i.z()++ )
         for( i.y() = begin.y(); i.y() < end.y(); i.y()++ )
            for( i.x() = begin.x(); i.x() < end.x(); i.x()++ )
               f( i, args... );
   }
};

template<>
struct ParallelFor3D< Devices::Host >
{
   template< typename MultiIndex, typename Function, typename... FunctionArgs >
   static void
   exec( MultiIndex begin, MultiIndex end, Devices::Host::LaunchConfiguration launch_config, Function f, FunctionArgs... args )
   {
      static_assert( MultiIndex::getSize() == 3, "ParallelFor3D requires a multi-index of size 3" );

#ifdef HAVE_OPENMP
      // Benchmarks show that this is significantly faster compared
      // to '#pragma omp parallel for if( Devices::Host::isOMPEnabled() )'
      if( Devices::Host::isOMPEnabled() ) {
         using Index = typename MultiIndex::ValueType;
         #pragma omp parallel for collapse( 2 )
         for( Index z = begin.z(); z < end.z(); z++ )
            for( Index y = begin.y(); y < end.y(); y++ ) {
               MultiIndex i{ begin.x(), y, z };
               for( ; i.x() < end.x(); i.x()++ )
                  f( i, args... );
            }
      }
      else {
         Devices::Sequential::LaunchConfiguration sequential_config;
         ParallelFor3D< Devices::Sequential >::exec( begin, end, sequential_config, f, args... );
      }
#else
      Devices::Sequential::LaunchConfiguration sequential_config;
      ParallelFor3D< Devices::Sequential >::exec( begin, end, sequential_config, f, args... );
#endif
   }
};

template< bool gridStride = true, typename MultiIndex, typename Function, typename... FunctionArgs >
__global__
void
ParallelFor3DKernel( MultiIndex begin, MultiIndex end, Function f, FunctionArgs... args )
{
#ifdef __CUDACC__
   // shift begin to the thread's initial position
   begin.x() += blockIdx.x * blockDim.x + threadIdx.x;
   begin.y() += blockIdx.y * blockDim.y + threadIdx.y;
   begin.z() += blockIdx.z * blockDim.z + threadIdx.z;

   // initialize iteration index
   MultiIndex i = begin;

   while( i.z() < end.z() ) {
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
      if( gridStride ) {
         i.y() = begin.y();
         i.z() += blockDim.z * gridDim.z;
      }
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
   template< typename MultiIndex, typename Function, typename... FunctionArgs >
   static void
   exec( MultiIndex begin, MultiIndex end, Devices::Cuda::LaunchConfiguration launch_config, Function f, FunctionArgs... args )
   {
      static_assert( MultiIndex::getSize() == 3, "ParallelFor3D requires a multi-index of size 3" );

      if( end.x() <= begin.x() || end.y() <= begin.y() || end.z() <= begin.z() )
         return;

      const auto sizeX = end.x() - begin.x();
      const auto sizeY = end.y() - begin.y();
      const auto sizeZ = end.z() - begin.z();

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
         constexpr auto kernel = ParallelFor3DKernel< false, MultiIndex, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
      }
      else {
         constexpr auto kernel = ParallelFor3DKernel< true, MultiIndex, Function, FunctionArgs... >;
         Cuda::launchKernel( kernel, launch_config, begin, end, f, args... );
      }
   }
};

}  // namespace TNL::Algorithms::detail
