// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <limits>       // std::numeric_limits
#include <memory>       // std::unique_ptr
#include <type_traits>  // std::remove_cv_t

#include "Types.h"
#include "Functions.h"
#include <TNL/Assert.h>
#include <TNL/DiscreteMath.h>
#include <TNL/Math.h>

namespace TNL::Backend {

constexpr std::size_t
getMaxGridXSize()
{
#if defined( __HIPCC__ )
   // AMD GPUs support up to 2^32 - 1 blocks in the x-dimension
   // https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html
   return 4294967295ULL;
#else
   return 2147483647;
#endif
}

constexpr std::size_t
getMaxGridYSize()
{
   // Both NVIDIA and AMD GPUs have the same limit of 65535 blocks
   // https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html
   return 65535;
}

constexpr std::size_t
getMaxGridZSize()
{
   // Both NVIDIA and AMD GPUs have the same limit of 65535 blocks
   // https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html
   return 65535;
}

constexpr std::size_t
getMaxThreadsPerGrid()
{
#if defined( __HIPCC__ )
   // AMD GPUs limit total threads per grid launch to 2^32 - 1
   // https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html
   return 4294967295ULL;
#else
   // NVIDIA GPUs have no explicit threads-per-grid limit beyond the grid/block
   // dimension limits
   return std::numeric_limits< std::size_t >::max();
#endif
}

constexpr int
getMaxThreadsPerBlock()
{
   // Both NVIDIA and AMD GPUs support up to 1024 threads per block
   return 1024;
}

/**
 * \brief Returns the maximal number of blocks of a one-dimensional grid with
 * `threadsPerBlock` threads per block.
 *
 * Besides the limit on the number of blocks, `getMaxGridXSize()`, it
 * respects the limit on the number of threads per grid,
 * `getMaxThreadsPerGrid()`, which AMD GPUs impose and HIP does not check.
 * For a multidimensional grid, pass the number of threads of one slice of the
 * grid in the x-dimension, i.e. the threads per block times the grid size in
 * the other dimensions.
 */
constexpr std::size_t
getMaxGridXSize( std::size_t threadsPerBlock )
{
   const std::size_t maxBlocks = getMaxThreadsPerGrid() / threadsPerBlock;
   // at least one block, so that loops over the grids always make progress;
   // an invalid configuration is then reported by the kernel launch
   if( maxBlocks == 0 )
      return 1;
   return maxBlocks < getMaxGridXSize() ? maxBlocks : getMaxGridXSize();
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

/**
 * \brief Returns the warp size for the target device.
 *
 * The `warpSize` variable is of type `int` and contains the warp size
 * (in threads) for the target device. This should be used only from device
 * code to develop portable wave-aware code.
 *
 * \note NVIDIA devices return 32; AMD devices return 64 for gfx8/gfx9 and
 * 32 for gfx10 and above.
 *
 * \warning The returned value may be inconsistent when used from a host-code
 * context on HIP, because `__GFX8__`/`__GFX9__` are defined only during
 * device compilation. On the host side, the function falls through to
 * return 32. Use `getMaxWarpSize()` for compile-time host-side guards,
 * or the runtime overload `getWarpSize(int deviceId)` for host-side
 * dispatch decisions. See
 * https://clang.llvm.org/docs/HIPSupport.html#predefined-macros for details.
 *
 * https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html#warpsize
 */
constexpr int
getWarpSize()
{
#if defined( __CUDACC__ ) || defined( __HIP_PLATFORM_NVCC__ )
   return 32;
// We use architecture macros instead of the deprecated `__AMDGCN_WAVEFRONT_SIZE__`
// which was removed in ROCm 7.0+.
#elif defined( __GFX8__ ) || defined( __GFX9__ )
   // gfx8/gfx9 (GCN) use wave64, all others use wave32
   return 64;
#else
   return 32;
#endif
}

/**
 * \brief Returns the maximum possible warp/wavefront size for the current build.
 *
 * Unlike getWarpSize(), this is consistent between host and device compilation:
 * - CUDA: always 32
 * - HIP: always 64 (the maximum across all AMD architectures)
 * - Host-only: 32
 *
 * Use this for compile-time guards that must be consistent on the host side,
 * e.g. deciding whether to instantiate TPS=64 kernel variants.
 */
constexpr int
getMaxWarpSize()
{
#if defined( __CUDACC__ ) || defined( __HIP_PLATFORM_NVCC__ )
   return 32;
#elif defined( __HIP__ )
   return 64;
#else
   return 32;
#endif
}

/**
 * \brief Returns the minimum possible warp/wavefront size across all platforms.
 *
 * Always returns 32. Use this for compile-time checks that must allow configs
 * valid on any architecture, e.g. column-major SlicedEllpack launch configs
 * where `TPS * SliceSize >= getMinWarpSize()` ensures the group covers at
 * least one warp on the smallest-warp-size device.
 */
constexpr int
getMinWarpSize()
{
   return 32;
}

/**
 * \brief Runtime query for the warp/wavefront size of a specific device.
 *
 * Use this for host-side launch configuration decisions (TPS selection, etc.).
 * For CUDA, always returns 32. For HIP, queries the runtime API.
 *
 * \param deviceId The device ID to query (default: current device).
 */
[[nodiscard]] inline int
getWarpSize( int deviceId )
{
#if defined( __CUDACC__ ) || defined( __HIP_PLATFORM_NVCC__ )
   (void) deviceId;
   return 32;
#elif defined( __HIP__ )
   int warpSize = 0;
   TNL_BACKEND_SAFE_CALL( hipDeviceGetAttribute( &warpSize, hipDeviceAttributeWarpSize, deviceId ) );
   return warpSize;
#else
   (void) deviceId;
   return 32;
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
 * \brief This function creates a buffer on the host, fills it with data
 * transferred from `source`, which is a pointer to device memory, and the
 * `push` handler processes the data in the buffer.
 */
template< typename Element, typename PushBuffer >
void
bufferedTransferToHost( const Element* source, std::size_t size, PushBuffer& push )
{
   using BufferType = std::remove_cv_t< Element >;
   auto fill = [ source ]( std::size_t offset, BufferType* buffer, std::size_t buffer_size )
   {
      Backend::memcpy(
         static_cast< void* >( buffer ),
         static_cast< const void* >( source + offset ),
         buffer_size * sizeof( Element ),
         Backend::MemcpyDeviceToHost );
   };
   bufferedTransfer< BufferType >( size, fill, push );
}

/**
 * \brief This function creates a buffer on the host, the `fill` handler fills
 * it with data and this function transfers data from the buffer to the
 * `destination`, which is a pointer to device memory.
 */
template< typename Element, typename FillBuffer >
void
bufferedTransferToDevice( Element* destination, std::size_t size, FillBuffer& fill )
{
   using BufferType = std::remove_cv_t< Element >;
   auto push = [ destination ]( std::size_t offset, const BufferType* buffer, std::size_t buffer_size, bool& next_iter )
   {
      Backend::memcpy(
         static_cast< void* >( destination + offset ),
         static_cast< const void* >( buffer ),
         buffer_size * sizeof( Element ),
         Backend::MemcpyHostToDevice );
   };
   bufferedTransfer< BufferType >( size, fill, push );
}

#if defined( __CUDACC__ ) || defined( __HIP__ )
/**
 * \brief Returns the global index of the current block in the x-dimension.
 *
 * Kernels launched as several grids get the index of the first block of their
 * grid, `blockOffset`, from the host as a kernel parameter.
 */
__device__
inline std::size_t
getGlobalBlockIdx_x( std::size_t blockOffset )
{
   return blockOffset + blockIdx.x;
}

//! \brief Returns the global index of the current block in the y-dimension, see \ref getGlobalBlockIdx_x.
__device__
inline std::size_t
getGlobalBlockIdx_y( std::size_t blockOffset )
{
   return blockOffset + blockIdx.y;
}

//! \brief Returns the global index of the current block in the z-dimension, see \ref getGlobalBlockIdx_x.
__device__
inline std::size_t
getGlobalBlockIdx_z( std::size_t blockOffset )
{
   return blockOffset + blockIdx.z;
}

//! \brief Returns the global index of the current thread in the x-dimension, see \ref getGlobalBlockIdx_x.
__device__
inline std::size_t
getGlobalThreadIdx_x( std::size_t blockOffset )
{
   return getGlobalBlockIdx_x( blockOffset ) * blockDim.x + threadIdx.x;
}

//! \brief Returns the global index of the current thread in the y-dimension, see \ref getGlobalBlockIdx_x.
__device__
inline std::size_t
getGlobalThreadIdx_y( std::size_t blockOffset )
{
   return getGlobalBlockIdx_y( blockOffset ) * blockDim.y + threadIdx.y;
}

//! \brief Returns the global index of the current thread in the z-dimension, see \ref getGlobalBlockIdx_x.
__device__
inline std::size_t
getGlobalThreadIdx_z( std::size_t blockOffset )
{
   return getGlobalBlockIdx_z( blockOffset ) * blockDim.z + threadIdx.z;
}
#endif

template< typename Index >
Index
getNumberOfBlocks( const Index threads, const unsigned int blockSize )
{
   // Index may be actually smaller type than unsigned int
   using CommonType = std::common_type_t< Index, unsigned int >;
   return roundUpDivision( static_cast< CommonType >( threads ), static_cast< CommonType >( blockSize ) );
}

template< typename Index >
Index
getNumberOfGrids( const Index blocks, const unsigned int gridSize )
{
   // Index may be actually smaller type than unsigned int
   using CommonType = std::common_type_t< Index, unsigned int >;
   return roundUpDivision( static_cast< CommonType >( blocks ), static_cast< CommonType >( gridSize ) );
}

/**
 * \brief Returns the number of blocks of the grid starting at the block
 * `blockOffset` when `totalBlocksCount` blocks are split into grids of at
 * most `maxGridSize` blocks.
 *
 * Typical use for a one-dimensional launch of `blocksCount` blocks:
 *
 * ```cpp
 * const std::size_t maxGridSize = Backend::getMaxGridXSize( launch_config.blockSize.x );
 * for( std::size_t blockOffset = 0; blockOffset < blocksCount; blockOffset += maxGridSize ) {
 *    launch_config.gridSize.x = Backend::getGridSize( blocksCount, blockOffset, maxGridSize );
 *    Backend::launchKernelAsync( kernel, launch_config, blockOffset, ... );
 * }
 * ```
 *
 * `blockOffset` must be smaller than `totalBlocksCount`.
 */
inline unsigned int
getGridSize( std::size_t totalBlocksCount, std::size_t blockOffset, std::size_t maxGridSize )
{
   TNL_ASSERT_LT( blockOffset, totalBlocksCount, "The block offset exceeds the number of blocks." );
   const std::size_t remainingBlocks = totalBlocksCount - blockOffset;
   return static_cast< unsigned int >( remainingBlocks < maxGridSize ? remainingBlocks : maxGridSize );
}

inline std::ostream&
operator<<( std::ostream& str, const dim3& d )
{
   str << "( " << d.x << ", " << d.y << ", " << d.z << " )";
   return str;
}

inline void
printThreadsSetup(
   const dim3& blockSize,
   const dim3& blocksCount,
   const dim3& gridSize,
   const dim3& gridsCount,
   std::ostream& str = std::cout )
{
   str << "Block size: " << blockSize << "\n Blocks count: " << blocksCount << "\n Grid size: " << gridSize
       << "\n Grids count: " << gridsCount << "\n";
}

}  // namespace TNL::Backend
