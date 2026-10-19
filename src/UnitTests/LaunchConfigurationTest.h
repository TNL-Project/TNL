#pragma once

#include <stdexcept>

#include <TNL/Backend/KernelLaunch.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/Containers/Array.h>

#include <gtest/gtest.h>

using namespace TNL;

// Returns a launch configuration with the given grid and block sizes.
inline Backend::LaunchConfiguration
makeConfiguration( dim3 gridSize, dim3 blockSize )
{
   return Backend::LaunchConfiguration( gridSize, blockSize );
}

TEST( LaunchConfigurationTest, validConfiguration )
{
   EXPECT_NO_THROW( Backend::checkLaunchConfiguration( makeConfiguration( dim3( 1 ), dim3( 1 ) ) ) );
   EXPECT_NO_THROW( Backend::checkLaunchConfiguration( makeConfiguration( dim3( 1000, 3, 2 ), dim3( 32, 32 ) ) ) );
   // the maximal grid size in the y and z dimensions (with few enough threads per grid for AMD GPUs)
   const dim3 maxBlock( Backend::getMaxThreadsPerBlock() );
   EXPECT_NO_THROW( Backend::checkLaunchConfiguration( makeConfiguration( dim3( 1, Backend::getMaxGridYSize() ), maxBlock ) ) );
   EXPECT_NO_THROW(
      Backend::checkLaunchConfiguration( makeConfiguration( dim3( 1, 1, Backend::getMaxGridZSize() ), maxBlock ) ) );
}

TEST( LaunchConfigurationTest, emptyGridOrBlock )
{
   EXPECT_THROW( Backend::checkLaunchConfiguration( makeConfiguration( dim3( 0 ), dim3( 256 ) ) ), std::invalid_argument );
   EXPECT_THROW( Backend::checkLaunchConfiguration( makeConfiguration( dim3( 1, 0 ), dim3( 256 ) ) ), std::invalid_argument );
   EXPECT_THROW(
      Backend::checkLaunchConfiguration( makeConfiguration( dim3( 1, 1, 0 ), dim3( 256 ) ) ), std::invalid_argument );
   EXPECT_THROW( Backend::checkLaunchConfiguration( makeConfiguration( dim3( 1 ), dim3( 0 ) ) ), std::invalid_argument );
   EXPECT_THROW( Backend::checkLaunchConfiguration( makeConfiguration( dim3( 1 ), dim3( 32, 0 ) ) ), std::invalid_argument );
}

TEST( LaunchConfigurationTest, tooManyBlocks )
{
   // the limit in the x-dimension is 2^32 - 1 on HIP, which does not fit into dim3 when increased by one
   if constexpr( Backend::getMaxGridXSize() < 4294967295ULL ) {
      const unsigned int gridSizeX = Backend::getMaxGridXSize() + 1;
      EXPECT_THROW(
         Backend::checkLaunchConfiguration( makeConfiguration( dim3( gridSizeX ), dim3( 1 ) ) ), std::invalid_argument );
   }
   const unsigned int gridSizeY = Backend::getMaxGridYSize() + 1;
   const unsigned int gridSizeZ = Backend::getMaxGridZSize() + 1;
   EXPECT_THROW(
      Backend::checkLaunchConfiguration( makeConfiguration( dim3( 1, gridSizeY ), dim3( 32 ) ) ), std::invalid_argument );
   EXPECT_THROW(
      Backend::checkLaunchConfiguration( makeConfiguration( dim3( 1, 1, gridSizeZ ), dim3( 32 ) ) ), std::invalid_argument );
}

TEST( LaunchConfigurationTest, blockTooLarge )
{
   const unsigned int blockSizeX = Backend::getMaxBlockXSize() + 1;
   const unsigned int blockSizeY = Backend::getMaxBlockYSize() + 1;
   const unsigned int blockSizeZ = Backend::getMaxBlockZSize() + 1;
   EXPECT_THROW(
      Backend::checkLaunchConfiguration( makeConfiguration( dim3( 4 ), dim3( blockSizeX ) ) ), std::invalid_argument );
   EXPECT_THROW(
      Backend::checkLaunchConfiguration( makeConfiguration( dim3( 4 ), dim3( 1, blockSizeY ) ) ), std::invalid_argument );
   EXPECT_THROW(
      Backend::checkLaunchConfiguration( makeConfiguration( dim3( 4 ), dim3( 1, 1, blockSizeZ ) ) ), std::invalid_argument );
}

TEST( LaunchConfigurationTest, tooManyThreadsPerBlock )
{
   // each dimension is within its limit, but the product exceeds the threads per block
   EXPECT_THROW( Backend::checkLaunchConfiguration( makeConfiguration( dim3( 4 ), dim3( 64, 32 ) ) ), std::invalid_argument );
}

TEST( LaunchConfigurationTest, threadsPerGrid )
{
   // 2^24 blocks with 256 threads, i.e. 2^32 threads in one grid
   const auto config = makeConfiguration( dim3( 1U << 24 ), dim3( 256 ) );
   if constexpr( Backend::getMaxThreadsPerGrid() < 4294967296ULL ) {
      // AMD GPUs limit the number of threads per grid to 2^32 - 1
      EXPECT_THROW( Backend::checkLaunchConfiguration( config ), std::invalid_argument );
   }
   else {
      EXPECT_NO_THROW( Backend::checkLaunchConfiguration( config ) );
   }
}

#if defined( __CUDACC__ ) || defined( __HIP__ )
// Counts the blocks in which the kernel runs.
__global__
void
countBlocksKernel( unsigned long long* counter )
{
   if( threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 )
      atomicAdd( counter, 1ULL );
}

TEST( LaunchConfigurationTest, launchKernel )
{
   Containers::Array< unsigned long long, Devices::GPU > counter( 1 );

   // a valid configuration runs all blocks
   counter.setValue( 0 );
   Backend::launchKernelSync( countBlocksKernel, makeConfiguration( dim3( 1000, 3 ), dim3( 32, 32 ) ), counter.getData() );
   EXPECT_EQ( counter.getElement( 0 ), 3000 );

   // an invalid configuration throws and the kernel is not launched
   counter.setValue( 0 );
   const unsigned int gridSizeY = Backend::getMaxGridYSize() + 1;
   EXPECT_THROW(
      Backend::launchKernelSync( countBlocksKernel, makeConfiguration( dim3( 1, gridSizeY ), dim3( 32 ) ), counter.getData() ),
      std::invalid_argument );
   EXPECT_EQ( counter.getElement( 0 ), 0 );

   if constexpr( Backend::getMaxThreadsPerGrid() < 4294967296ULL ) {
      // HIP would silently run fewer threads (no thread at all for 2^32 threads)
      EXPECT_THROW(
         Backend::launchKernelSync( countBlocksKernel, makeConfiguration( dim3( 1U << 24 ), dim3( 256 ) ), counter.getData() ),
         std::invalid_argument );
      EXPECT_EQ( counter.getElement( 0 ), 0 );
   }
}
#endif
