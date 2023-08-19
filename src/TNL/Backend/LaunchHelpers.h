// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>

#include "Types.h"
#include <TNL/DiscreteMath.h>
#include <TNL/Math.h>

namespace TNL::Backend {

inline constexpr std::size_t
getMaxGridXSize()
{
   return 2147483647;
}

inline constexpr std::size_t
getMaxGridYSize()
{
#if defined( __CUDACC__ ) || defined( __HIP_PLATFORM_NVCC__ )
   return 65535;
#else
   return 2147483647;
#endif
}

inline constexpr std::size_t
getMaxGridZSize()
{
#if defined( __CUDACC__ ) || defined( __HIP_PLATFORM_NVCC__ )
   return 65535;
#else
   return 2147483647;
#endif
}

inline constexpr int
getMaxBlockXSize()
{
   return 1024;
}

inline constexpr int
getMaxBlockYSize()
{
   return 1024;
}

inline constexpr int
getMaxBlockZSize()
{
#if defined( __CUDACC__ ) || defined( __HIP_PLATFORM_NVCC__ )
   return 64;
#else
   return 1024;
#endif
}

/*
 * The warpSize variable is of type int and contains the warp size (in threads)
 * for the target device. Note that all current Nvidia devices return 32 for
 * this variable, and all current AMD devices return 64. Device code should use
 * the warpSize built-in to develop portable wave-aware code.
 *
 * https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html
 */
inline constexpr int
getWarpSize()
{
#if defined( __CUDACC__ ) || defined( __HIP_PLATFORM_NVCC__ )
   return 32;
#else
   return 64;
#endif
}

// When we transfer data between the GPU and the CPU we use 1 MiB buffer. This
// size should ensure good performance.
// We use the same buffer size even for retyping data during IO operations.
inline constexpr int
getTransferBufferSize()
{
   return 1 << 20;
}

#if defined( __CUDACC__ ) || defined( __HIP__ )
__device__
inline int
getGlobalThreadIdx_x( const dim3& gridIdx )
{
   return ( gridIdx.x * getMaxGridXSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
}

__device__
inline int
getGlobalThreadIdx_y( const dim3& gridIdx )
{
   return ( gridIdx.y * getMaxGridYSize() + blockIdx.y ) * blockDim.y + threadIdx.y;
}

__device__
inline int
getGlobalThreadIdx_z( const dim3& gridIdx )
{
   return ( gridIdx.z * getMaxGridZSize() + blockIdx.z ) * blockDim.z + threadIdx.z;
}
#endif

inline int
getNumberOfBlocks( const int threads, const int blockSize )
{
   return roundUpDivision( threads, blockSize );
}

inline int
getNumberOfGrids( const int blocks, const int gridSize )
{
   return roundUpDivision( blocks, gridSize );
}

inline void
setupThreads( const dim3& blockSize,
              dim3& blocksCount,
              dim3& gridsCount,
              long long int xThreads,
              long long int yThreads = 0,
              long long int zThreads = 0 )
{
   blocksCount.x = max( 1, xThreads / blockSize.x + static_cast< long long int >( xThreads % blockSize.x != 0 ) );
   blocksCount.y = max( 1, yThreads / blockSize.y + static_cast< long long int >( yThreads % blockSize.y != 0 ) );
   blocksCount.z = max( 1, zThreads / blockSize.z + static_cast< long long int >( zThreads % blockSize.z != 0 ) );

   gridsCount.x =
      blocksCount.x / getMaxGridXSize() + static_cast< unsigned long int >( blocksCount.x % getMaxGridXSize() != 0 );
   gridsCount.y =
      blocksCount.y / getMaxGridYSize() + static_cast< unsigned long int >( blocksCount.y % getMaxGridYSize() != 0 );
   gridsCount.z =
      blocksCount.z / getMaxGridZSize() + static_cast< unsigned long int >( blocksCount.z % getMaxGridZSize() != 0 );
}

inline void
setupGrid( const dim3& blocksCount, const dim3& gridsCount, const dim3& gridIdx, dim3& gridSize )
{
   if( gridIdx.x < gridsCount.x - 1 )
      gridSize.x = getMaxGridXSize();
   else
      gridSize.x = blocksCount.x % getMaxGridXSize();

   if( gridIdx.y < gridsCount.y - 1 )
      gridSize.y = getMaxGridYSize();
   else
      gridSize.y = blocksCount.y % getMaxGridYSize();

   if( gridIdx.z < gridsCount.z - 1 )
      gridSize.z = getMaxGridZSize();
   else
      gridSize.z = blocksCount.z % getMaxGridZSize();
}

inline std::ostream&
operator<<( std::ostream& str, const dim3& d )
{
   str << "( " << d.x << ", " << d.y << ", " << d.z << " )";
   return str;
}

inline void
printThreadsSetup( const dim3& blockSize,
                   const dim3& blocksCount,
                   const dim3& gridSize,
                   const dim3& gridsCount,
                   std::ostream& str = std::cout )
{
   str << "Block size: " << blockSize << std::endl
       << " Blocks count: " << blocksCount << std::endl
       << " Grid size: " << gridSize << std::endl
       << " Grids count: " << gridsCount << std::endl;
}

}  // namespace TNL::Backend
