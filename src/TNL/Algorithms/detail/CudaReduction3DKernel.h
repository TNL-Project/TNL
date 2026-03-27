// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Backend.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/CudaReductionBuffer.h>
#include "CudaReductionKernel.h"

namespace TNL::Algorithms::detail {

template< int blockSizeX, typename Result, typename DataFetcher, typename Reduction, typename Index >
__global__
void
CudaReduction3DKernel( const Result identity,
                       DataFetcher dataFetcher,
                       const Reduction reduction,
                       const Index size,
                       const int m,
                       const int n,
                       Result* output )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   TNL_ASSERT_EQ( blockDim.x, blockSizeX, "unexpected block size in CudaReduction2DKernel" );

   // allocate shared memory
   using BlockReduce = CudaBlockReduceSharedMemory< blockSizeX, Reduction, Result >;
   __shared__ typename BlockReduce::Storage blockReduceStorage;

   // Get the thread id (tid), global thread id (gid) and gridSize.
   const Index tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
   Index gid = blockIdx.x * blockDim.x + threadIdx.x;
   const Index gridSizeX = blockDim.x * gridDim.x;

   // Get the dataset index.
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   const int z = blockIdx.z * blockDim.z + threadIdx.z;
   if( y >= m || z >= n )
      return;

   // Start with the sequential reduction and push the result into the shared memory.
   Result result = identity;
   while( gid + 4 * gridSizeX < size ) {
      result = reduction( result, dataFetcher( gid, y, z ) );
      result = reduction( result, dataFetcher( gid + gridSizeX, y, z ) );
      result = reduction( result, dataFetcher( gid + 2 * gridSizeX, y, z ) );
      result = reduction( result, dataFetcher( gid + 3 * gridSizeX, y, z ) );
      gid += 4 * gridSizeX;
   }
   while( gid + 2 * gridSizeX < size ) {
      result = reduction( result, dataFetcher( gid, y, z ) );
      result = reduction( result, dataFetcher( gid + gridSizeX, y, z ) );
      gid += 2 * gridSizeX;
   }
   while( gid < size ) {
      result = reduction( result, dataFetcher( gid, y, z ) );
      gid += gridSizeX;
   }
   __syncthreads();

   // Perform the parallel reduction.
   result = BlockReduce::reduce( reduction, identity, result, blockReduceStorage, tid, threadIdx.x, tid );

   // Store the result back in the global memory.
   if( threadIdx.x == 0 ) {
      output[ blockIdx.x + z * gridDim.x + y * n * gridDim.x ] = result;
   }
#endif
}

template< typename Result, typename DataFetcher, typename Reduction, typename Index >
dim3
CudaReduction3DKernelLauncher( const Result identity,
                               DataFetcher dataFetcher,
                               const Reduction reduction,
                               const Index size,
                               const int m,
                               const int n,
                               Result*& output )
{
   // must be a power of 2
   static constexpr int maxThreadsPerBlock = 512;

   const int activeDevice = Backend::getDevice();
   const int desGridSizeX = Backend::getDeviceMultiprocessors( activeDevice );
   Backend::LaunchConfiguration launch_config;

   launch_config.blockSize.y = TNL::min( m, 4 );
   launch_config.blockSize.z = TNL::min( n, 4 );

   // launch_config.blockSize.x has to be a power of 2
   launch_config.blockSize.x = maxThreadsPerBlock;
   while( launch_config.blockSize.x * launch_config.blockSize.y * launch_config.blockSize.z > maxThreadsPerBlock )
      launch_config.blockSize.x /= 2;

   launch_config.gridSize.x = TNL::min( Backend::getNumberOfBlocks( size, launch_config.blockSize.x ), desGridSizeX );
   launch_config.gridSize.y = Backend::getNumberOfBlocks( m, launch_config.blockSize.y );
   launch_config.gridSize.z = Backend::getNumberOfBlocks( n, launch_config.blockSize.z );

   if( launch_config.gridSize.y > Backend::getMaxGridYSize() ) {
      throw std::logic_error( "Maximum launch_config.gridSize.y limit exceeded (limit is "
                              + std::to_string( Backend::getMaxGridYSize() ) + ", attempted "
                              + std::to_string( launch_config.gridSize.y ) + ")." );
   }
   if( launch_config.gridSize.z > Backend::getMaxGridZSize() ) {
      throw std::logic_error( "Maximum launch_config.gridSize.z limit exceeded (limit is "
                              + std::to_string( Backend::getMaxGridZSize() ) + ", attempted "
                              + std::to_string( launch_config.gridSize.z ) + ")." );
   }

   // create reference to the reduction buffer singleton and set size
   const std::size_t buf_size = 8 * ( ( m * n ) / 8 + 1 ) * desGridSizeX * sizeof( Result );
   CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
   cudaReductionBuffer.setSize( buf_size );
   output = cudaReductionBuffer.template getData< Result >();

   // when there is only one warp per launch_config.blockSize.x, we need to allocate two warps
   // worth of shared memory so that we don't index shared memory out of bounds
   launch_config.dynamicSharedMemorySize =
      ( launch_config.blockSize.x <= 32 )
         ? 2 * launch_config.blockSize.x * launch_config.blockSize.y * launch_config.blockSize.z * sizeof( Result )
         : launch_config.blockSize.x * launch_config.blockSize.y * launch_config.blockSize.z * sizeof( Result );

   // Depending on the blockSize we generate appropriate template instance.
   switch( launch_config.blockSize.x ) {
      case 512:
         Backend::launchKernelSync( CudaReduction3DKernel< 512, Result, DataFetcher, Reduction, Index >,
                                    launch_config,
                                    identity,
                                    dataFetcher,
                                    reduction,
                                    size,
                                    m,
                                    n,
                                    output );
         break;
      case 256:
         Backend::funcSetCacheConfig( CudaReduction3DKernel< 256, Result, DataFetcher, Reduction, Index >,
                                      Backend::FuncCachePreferShared );
         Backend::launchKernelSync( CudaReduction3DKernel< 256, Result, DataFetcher, Reduction, Index >,
                                    launch_config,
                                    identity,
                                    dataFetcher,
                                    reduction,
                                    size,
                                    m,
                                    n,
                                    output );
         break;
      case 128:
         Backend::funcSetCacheConfig( CudaReduction3DKernel< 128, Result, DataFetcher, Reduction, Index >,
                                      Backend::FuncCachePreferShared );
         Backend::launchKernelSync( CudaReduction3DKernel< 128, Result, DataFetcher, Reduction, Index >,
                                    launch_config,
                                    identity,
                                    dataFetcher,
                                    reduction,
                                    size,
                                    m,
                                    n,
                                    output );
         break;
      case 64:
         Backend::funcSetCacheConfig( CudaReduction3DKernel< 64, Result, DataFetcher, Reduction, Index >,
                                      Backend::FuncCachePreferShared );
         Backend::launchKernelSync( CudaReduction3DKernel< 64, Result, DataFetcher, Reduction, Index >,
                                    launch_config,
                                    identity,
                                    dataFetcher,
                                    reduction,
                                    size,
                                    m,
                                    n,
                                    output );
         break;
      case 32:
         Backend::funcSetCacheConfig( CudaReduction3DKernel< 32, Result, DataFetcher, Reduction, Index >,
                                      Backend::FuncCachePreferShared );
         Backend::launchKernelSync( CudaReduction3DKernel< 32, Result, DataFetcher, Reduction, Index >,
                                    launch_config,
                                    identity,
                                    dataFetcher,
                                    reduction,
                                    size,
                                    m,
                                    n,
                                    output );
         break;
      case 16:
         Backend::funcSetCacheConfig( CudaReduction3DKernel< 16, Result, DataFetcher, Reduction, Index >,
                                      Backend::FuncCachePreferShared );
         Backend::launchKernelSync( CudaReduction3DKernel< 16, Result, DataFetcher, Reduction, Index >,
                                    launch_config,
                                    identity,
                                    dataFetcher,
                                    reduction,
                                    size,
                                    m,
                                    n,
                                    output );
         break;
      case 8:
         Backend::funcSetCacheConfig( CudaReduction3DKernel< 8, Result, DataFetcher, Reduction, Index >,
                                      Backend::FuncCachePreferShared );
         Backend::launchKernelSync( CudaReduction3DKernel< 8, Result, DataFetcher, Reduction, Index >,
                                    launch_config,
                                    identity,
                                    dataFetcher,
                                    reduction,
                                    size,
                                    m,
                                    n,
                                    output );
         break;
      case 4:
         Backend::funcSetCacheConfig( CudaReduction3DKernel< 4, Result, DataFetcher, Reduction, Index >,
                                      Backend::FuncCachePreferShared );
         Backend::launchKernelSync( CudaReduction3DKernel< 4, Result, DataFetcher, Reduction, Index >,
                                    launch_config,
                                    identity,
                                    dataFetcher,
                                    reduction,
                                    size,
                                    m,
                                    n,
                                    output );
         break;
      case 2:
         Backend::funcSetCacheConfig( CudaReduction3DKernel< 2, Result, DataFetcher, Reduction, Index >,
                                      Backend::FuncCachePreferShared );
         Backend::launchKernelSync( CudaReduction3DKernel< 2, Result, DataFetcher, Reduction, Index >,
                                    launch_config,
                                    identity,
                                    dataFetcher,
                                    reduction,
                                    size,
                                    m,
                                    n,
                                    output );
         break;
      case 1:
         throw std::logic_error( "blockSize should not be 1." );
      default:
         throw std::logic_error( "Block size is " + std::to_string( launch_config.blockSize.x )
                                 + " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
   }

   return launch_config.gridSize;
}

}  // namespace TNL::Algorithms::detail
