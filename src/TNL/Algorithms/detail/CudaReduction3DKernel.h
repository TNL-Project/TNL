// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Backend.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/CudaReductionBuffer.h>

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
   // Create a shared memory buffer for the reduction.
   Result* sdata = Backend::getSharedMemory< Result >();

   // Get the thread id (tid), global thread id (gid) and gridSize.
   const Index tid = threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
   Index gid = blockIdx.x * blockDim.x + threadIdx.x;
   const Index gridSizeX = blockDim.x * gridDim.x;

   // Get the dataset index.
   const int y = blockIdx.y * blockDim.y + threadIdx.y;
   const int z = blockIdx.z * blockDim.z + threadIdx.z;
   if( y >= m || z >= n )
      return;

   sdata[ tid ] = identity;

   // Start with the sequential reduction and push the result into the shared memory.
   while( gid + 4 * gridSizeX < size ) {
      sdata[ tid ] = reduction( sdata[ tid ], dataFetcher( gid, y, z ) );
      sdata[ tid ] = reduction( sdata[ tid ], dataFetcher( gid + gridSizeX, y, z ) );
      sdata[ tid ] = reduction( sdata[ tid ], dataFetcher( gid + 2 * gridSizeX, y, z ) );
      sdata[ tid ] = reduction( sdata[ tid ], dataFetcher( gid + 3 * gridSizeX, y, z ) );
      gid += 4 * gridSizeX;
   }
   while( gid + 2 * gridSizeX < size ) {
      sdata[ tid ] = reduction( sdata[ tid ], dataFetcher( gid, y, z ) );
      sdata[ tid ] = reduction( sdata[ tid ], dataFetcher( gid + gridSizeX, y, z ) );
      gid += 2 * gridSizeX;
   }
   while( gid < size ) {
      sdata[ tid ] = reduction( sdata[ tid ], dataFetcher( gid, y, z ) );
      gid += gridSizeX;
   }
   __syncthreads();

   // Perform the parallel reduction.
   if( blockSizeX >= 1024 ) {
      if( threadIdx.x < 512 )
         sdata[ tid ] = reduction( sdata[ tid ], sdata[ tid + 512 ] );
      __syncthreads();
   }
   if( blockSizeX >= 512 ) {
      if( threadIdx.x < 256 )
         sdata[ tid ] = reduction( sdata[ tid ], sdata[ tid + 256 ] );
      __syncthreads();
   }
   if( blockSizeX >= 256 ) {
      if( threadIdx.x < 128 )
         sdata[ tid ] = reduction( sdata[ tid ], sdata[ tid + 128 ] );
      __syncthreads();
   }
   if( blockSizeX >= 128 ) {
      if( threadIdx.x < 64 )
         sdata[ tid ] = reduction( sdata[ tid ], sdata[ tid + 64 ] );
      __syncthreads();
   }
   if( threadIdx.x < 32 ) {
      // Fetch final intermediate sum from 2nd warp.
      if( blockSizeX >= 64 )
         sdata[ tid ] = reduction( sdata[ tid ], sdata[ tid + 32 ] );
      __syncwarp();
      // Note that here we do not have to check if tid < 16 etc, because we have
      // 2 * blockSize.x elements of shared memory per block, so we do not
      // access out of bounds. The results for the upper half will be undefined,
      // but unused anyway.
      if( blockSizeX >= 32 )
         sdata[ tid ] = reduction( sdata[ tid ], sdata[ tid + 16 ] );
      __syncwarp();
      if( blockSizeX >= 16 )
         sdata[ tid ] = reduction( sdata[ tid ], sdata[ tid + 8 ] );
      __syncwarp();
      if( blockSizeX >= 8 )
         sdata[ tid ] = reduction( sdata[ tid ], sdata[ tid + 4 ] );
      __syncwarp();
      if( blockSizeX >= 4 )
         sdata[ tid ] = reduction( sdata[ tid ], sdata[ tid + 2 ] );
      __syncwarp();
      if( blockSizeX >= 2 )
         sdata[ tid ] = reduction( sdata[ tid ], sdata[ tid + 1 ] );
   }
   // Store the result back in the global memory.
   if( threadIdx.x == 0 ) {
      output[ blockIdx.x + z * gridDim.x + y * n * gridDim.x ] = sdata[ tid ];
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

   if( launch_config.gridSize.y > (unsigned) Backend::getMaxGridYSize() ) {
      throw std::logic_error( "Maximum launch_config.gridSize.y limit exceeded (limit is "
                              + std::to_string( Backend::getMaxGridYSize() ) + ", attempted "
                              + std::to_string( launch_config.gridSize.y ) + ")." );
   }
   if( launch_config.gridSize.z > (unsigned) Backend::getMaxGridZSize() ) {
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
