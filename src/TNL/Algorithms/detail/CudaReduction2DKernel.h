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
CudaReduction2DKernel(
   const Result identity,
   DataFetcher dataFetcher,
   const Reduction reduction,
   const Index size,
   const int n,
   Result* output )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   TNL_ASSERT_EQ( blockDim.x, blockSizeX, "unexpected block size in CudaReduction2DKernel" );

   // allocate shared memory
   using BlockReduce = CudaBlockReduceSharedMemory< blockSizeX, Reduction, Result >;
   __shared__ typename BlockReduce::Storage blockReduceStorage;

   // Get the thread id (tid), global thread id (gid) and gridSize.
   const Index tid = threadIdx.y * blockDim.x + threadIdx.x;
   Index gid = blockIdx.x * blockDim.x + threadIdx.x;
   const Index gridSizeX = blockDim.x * gridDim.x;

   // Get the dataset index.
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   if( y >= n )
      return;

   // Start with the sequential reduction and push the result into the shared memory.
   Result result = identity;
   while( gid + 4 * gridSizeX < size ) {
      result = reduction( result, dataFetcher( gid, y ) );
      result = reduction( result, dataFetcher( gid + gridSizeX, y ) );
      result = reduction( result, dataFetcher( gid + 2 * gridSizeX, y ) );
      result = reduction( result, dataFetcher( gid + 3 * gridSizeX, y ) );
      gid += 4 * gridSizeX;
   }
   while( gid + 2 * gridSizeX < size ) {
      result = reduction( result, dataFetcher( gid, y ) );
      result = reduction( result, dataFetcher( gid + gridSizeX, y ) );
      gid += 2 * gridSizeX;
   }
   while( gid < size ) {
      result = reduction( result, dataFetcher( gid, y ) );
      gid += gridSizeX;
   }
   __syncthreads();

   // Perform the parallel reduction.
   result = BlockReduce::reduce( reduction, identity, result, blockReduceStorage, tid, threadIdx.x, tid );

   // Store the result back in the global memory.
   if( threadIdx.x == 0 ) {
      output[ blockIdx.x + y * gridDim.x ] = result;
   }
#endif
}

template< typename Result, typename DataFetcher, typename Reduction, typename Index >
int
CudaReduction2DKernelLauncher(
   const Result identity,
   DataFetcher dataFetcher,
   const Reduction reduction,
   const Index size,
   const int n,
   Result*& output )
{
   // must be a power of 2
   static constexpr int maxThreadsPerBlock = 256;

   // The number of blocks should be a multiple of the number of multiprocessors
   // to ensure optimum balancing of the load. This is very important, because
   // we run the kernel with a fixed number of blocks, so the amount of work per
   // block increases with enlarging the problem, so even small imbalance can
   // cost us dearly.
   // Therefore,  desGridSize = blocksPerMultiprocessor * numberOfMultiprocessors
   // where the maximum value of blocksPerMultiprocessor can be determined
   // according to the number of available registers on the multiprocessor.
   // However, it seems to be better to map only one CUDA block per multiprocessor,
   // or maybe just slightly more.
   const int activeDevice = Backend::getDevice();
   const int desGridSizeX = Backend::getDeviceMultiprocessors( activeDevice );
   Backend::LaunchConfiguration launch_config;

   // version A: max 16 rows of threads
   launch_config.blockSize.y = TNL::min( n, 16 );

   // version B: up to 16 rows of threads, then "minimize" number of inactive rows
   //   if( n <= 16 )
   //      launch_config.blockSize.y = n;
   //   else {
   //      int r = (n - 1) % 16 + 1;
   //      if( r > 12 )
   //         launch_config.blockSize.y = 16;
   //      else if( r > 8 )
   //         launch_config.blockSize.y = 4;
   //      else if( r > 4 )
   //         launch_config.blockSize.y = 8;
   //      else
   //         launch_config.blockSize.y = 4;
   //   }

   // launch_config.blockSize.x has to be a power of 2
   launch_config.blockSize.x = maxThreadsPerBlock;
   while( launch_config.blockSize.x * launch_config.blockSize.y > maxThreadsPerBlock )
      launch_config.blockSize.x /= 2;

   launch_config.gridSize.x = TNL::min( Backend::getNumberOfBlocks( size, launch_config.blockSize.x ), desGridSizeX );
   launch_config.gridSize.y = Backend::getNumberOfBlocks( n, launch_config.blockSize.y );

   if( launch_config.gridSize.y > Backend::getMaxGridYSize() ) {
      throw std::logic_error(
         "Maximum launch_config.gridSize.y limit exceeded (limit is " + std::to_string( Backend::getMaxGridYSize() )
         + ", attempted " + std::to_string( launch_config.gridSize.y ) + ")." );
   }

   // create reference to the reduction buffer singleton and set size
   // (make an overestimate to avoid reallocation on every call if n is increased by 1 each time)
   const std::size_t buf_size = 8 * ( n / 8 + 1 ) * desGridSizeX * sizeof( Result );
   CudaReductionBuffer& cudaReductionBuffer = CudaReductionBuffer::getInstance();
   cudaReductionBuffer.setSize( buf_size );
   output = cudaReductionBuffer.template getData< Result >();

   // when there is only one warp per launch_config.blockSize.x, we need to allocate two warps
   // worth of shared memory so that we don't index shared memory out of bounds
   launch_config.dynamicSharedMemorySize = ( launch_config.blockSize.x <= 32 )
                                            ? 2 * launch_config.blockSize.x * launch_config.blockSize.y * sizeof( Result )
                                            : launch_config.blockSize.x * launch_config.blockSize.y * sizeof( Result );

   // Depending on the blockSize we generate appropriate template instance.
   switch( launch_config.blockSize.x ) {
      case 512:
         Backend::launchKernelSync(
            CudaReduction2DKernel< 512, Result, DataFetcher, Reduction, Index >,
            launch_config,
            identity,
            dataFetcher,
            reduction,
            size,
            n,
            output );
         break;
      case 256:
         Backend::funcSetCacheConfig(
            CudaReduction2DKernel< 256, Result, DataFetcher, Reduction, Index >, Backend::FuncCachePreferShared );
         Backend::launchKernelSync(
            CudaReduction2DKernel< 256, Result, DataFetcher, Reduction, Index >,
            launch_config,
            identity,
            dataFetcher,
            reduction,
            size,
            n,
            output );
         break;
      case 128:
         Backend::funcSetCacheConfig(
            CudaReduction2DKernel< 128, Result, DataFetcher, Reduction, Index >, Backend::FuncCachePreferShared );
         Backend::launchKernelSync(
            CudaReduction2DKernel< 128, Result, DataFetcher, Reduction, Index >,
            launch_config,
            identity,
            dataFetcher,
            reduction,
            size,
            n,
            output );
         break;
      case 64:
         Backend::funcSetCacheConfig(
            CudaReduction2DKernel< 64, Result, DataFetcher, Reduction, Index >, Backend::FuncCachePreferShared );
         Backend::launchKernelSync(
            CudaReduction2DKernel< 64, Result, DataFetcher, Reduction, Index >,
            launch_config,
            identity,
            dataFetcher,
            reduction,
            size,
            n,
            output );
         break;
      case 32:
         Backend::funcSetCacheConfig(
            CudaReduction2DKernel< 32, Result, DataFetcher, Reduction, Index >, Backend::FuncCachePreferShared );
         Backend::launchKernelSync(
            CudaReduction2DKernel< 32, Result, DataFetcher, Reduction, Index >,
            launch_config,
            identity,
            dataFetcher,
            reduction,
            size,
            n,
            output );
         break;
      case 16:
         Backend::funcSetCacheConfig(
            CudaReduction2DKernel< 16, Result, DataFetcher, Reduction, Index >, Backend::FuncCachePreferShared );
         Backend::launchKernelSync(
            CudaReduction2DKernel< 16, Result, DataFetcher, Reduction, Index >,
            launch_config,
            identity,
            dataFetcher,
            reduction,
            size,
            n,
            output );
         break;
      case 8:
         Backend::funcSetCacheConfig(
            CudaReduction2DKernel< 8, Result, DataFetcher, Reduction, Index >, Backend::FuncCachePreferShared );
         Backend::launchKernelSync(
            CudaReduction2DKernel< 8, Result, DataFetcher, Reduction, Index >,
            launch_config,
            identity,
            dataFetcher,
            reduction,
            size,
            n,
            output );
         break;
      case 4:
         Backend::funcSetCacheConfig(
            CudaReduction2DKernel< 4, Result, DataFetcher, Reduction, Index >, Backend::FuncCachePreferShared );
         Backend::launchKernelSync(
            CudaReduction2DKernel< 4, Result, DataFetcher, Reduction, Index >,
            launch_config,
            identity,
            dataFetcher,
            reduction,
            size,
            n,
            output );
         break;
      case 2:
         Backend::funcSetCacheConfig(
            CudaReduction2DKernel< 2, Result, DataFetcher, Reduction, Index >, Backend::FuncCachePreferShared );
         Backend::launchKernelSync(
            CudaReduction2DKernel< 2, Result, DataFetcher, Reduction, Index >,
            launch_config,
            identity,
            dataFetcher,
            reduction,
            size,
            n,
            output );
         break;
      case 1:
         throw std::logic_error( "blockSize should not be 1." );
      default:
         throw std::logic_error(
            "Block size is " + std::to_string( launch_config.blockSize.x )
            + " which is none of 1, 2, 4, 8, 16, 32, 64, 128, 256 or 512." );
   }

   // return the size of the output array on the CUDA device
   return launch_config.gridSize.x;
}

}  // namespace TNL::Algorithms::detail
