// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <memory>       // std::unique_ptr
#include <type_traits>  // std::remove_cv_t

#include "Types.h"
#include "Functions.h"
#include <TNL/DiscreteMath.h>
#include <TNL/Math.h>

namespace TNL::Backend {

constexpr std::size_t
getMaxGridXSize()
{
   return 2147483647;
}

constexpr std::size_t
getMaxGridYSize()
{
#if defined( __CUDACC__ ) || defined( __HIP_PLATFORM_NVCC__ )
   return 65535;
#else
   return 2147483647;
#endif
}

constexpr std::size_t
getMaxGridZSize()
{
#if defined( __CUDACC__ ) || defined( __HIP_PLATFORM_NVCC__ )
   return 65535;
#else
   return 2147483647;
#endif
}

constexpr int
getMaxBlockXSize()
{
   return 1024;
}

constexpr int
getMaxBlockYSize()
{
   return 1024;
}

constexpr int
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
 * for the target device. This should be used only from device code in order to
 * develop portable wave-aware code.
 *
 * Note that NVIDIA devices return 32; AMD devices return 64 for gfx9 and 32 for
 * gfx10 and above.
 *
 * Warning: the returned value may be inconsistent when used from a host-code
 * context on HIP where it defaults to 64, but the device code may use 32.
 * See https://clang.llvm.org/docs/HIPSupport.html#predefined-macros for details.
 *
 * https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#warpsize
 */
constexpr int
getWarpSize()
{
#if defined( __CUDACC__ ) || defined( __HIP_PLATFORM_NVCC__ )
   return 32;
#elif defined( __AMDGCN_WAVEFRONT_SIZE__ )
   return __AMDGCN_WAVEFRONT_SIZE__;
// this is deprecated, alias to the previous one
#elif defined( __AMDGCN_WAVEFRONT_SIZE )
   return __AMDGCN_WAVEFRONT_SIZE;
#else
   return 64;  // default, same as __AMDGCN_WAVEFRONT_SIZE__ for host code
#endif
}

// When we transfer data between the GPU and the CPU we use 1 MiB buffer. This
// size should ensure good performance.
// We use the same buffer size even for retyping data during IO operations.
constexpr std::size_t
getTransferBufferSize()
{
   return 1 << 20;
}

template< typename Element, typename FillBuffer, typename PushBuffer >
void
bufferedTransfer( std::size_t size, FillBuffer& fill, PushBuffer& push )
{
   const std::size_t buffer_size = std::min( Backend::getTransferBufferSize() / sizeof( Element ), size );
   std::unique_ptr< Element[] > host_buffer{ new Element[ buffer_size ] };

   std::size_t compared = 0;
   while( compared < size ) {
      const std::size_t transfer = std::min( size - compared, buffer_size );
      fill( compared, host_buffer.get(), transfer );
      bool next_iter = true;
      push( compared, host_buffer.get(), transfer, next_iter );
      if( ! next_iter )
         return;
      compared += transfer;
   }
}

/**
 * This function creates a buffer on the host, fills it with data transferred
 * from \e source, which is a pointer to device memory, and the \e push handler
 * processes the data in the buffer.
 */
template< typename Element, typename PushBuffer >
void
bufferedTransferToHost( const Element* source, std::size_t size, PushBuffer& push )
{
   using BufferType = std::remove_cv_t< Element >;
   auto fill = [ source ]( std::size_t offset, BufferType* buffer, std::size_t buffer_size )
   {
      Backend::memcpy( static_cast< void* >( buffer ),
                       static_cast< const void* >( source + offset ),
                       buffer_size * sizeof( Element ),
                       Backend::MemcpyDeviceToHost );
   };
   bufferedTransfer< BufferType >( size, fill, push );
}

/**
 * This function creates a buffer on the host, the \e fill handler fills it
 * with data and this function transfers data from the buffer to the
 * \e destination, which is a pointer to device memory.
 */
template< typename Element, typename FillBuffer >
void
bufferedTransferToDevice( Element* destination, std::size_t size, FillBuffer& fill )
{
   using BufferType = std::remove_cv_t< Element >;
   auto push = [ destination ]( std::size_t offset, const BufferType* buffer, std::size_t buffer_size, bool& next_iter )
   {
      Backend::memcpy( static_cast< void* >( destination + offset ),
                       static_cast< const void* >( buffer ),
                       buffer_size * sizeof( Element ),
                       Backend::MemcpyHostToDevice );
   };
   bufferedTransfer< BufferType >( size, fill, push );
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

__device__
inline int
getGlobalBlockIdx_x( const dim3& gridIdx )
{
   return ( gridIdx.x * getMaxGridXSize() + blockIdx.x );
}

__device__
inline int
getGlobalBlockIdx_y( const dim3& gridIdx )
{
   return ( gridIdx.y * getMaxGridYSize() + blockIdx.y );
}

__device__
inline int
getGlobalBlockIdx_z( const dim3& gridIdx )
{
   return ( gridIdx.z * getMaxGridZSize() + blockIdx.z );
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
   str << "Block size: " << blockSize << "\n Blocks count: " << blocksCount << "\n Grid size: " << gridSize
       << "\n Grids count: " << gridsCount << "\n";
}

}  // namespace TNL::Backend
