#pragma once

// AdaptiveCSR-specific tests, included in the generated ReduceSegmentsTest_AdaptiveCSR sources
// (see tnl_add_format_tests in CMakeLists.txt)

#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Containers/Vector.h>

#include <gtest/gtest.h>

// The blocks of AdaptiveCSR are created only on GPUs, on the other devices the
// reduction falls back to the CSR kernels.
template< typename Index >
void
test_AdaptiveCSR_blocks()
{
   using HostSegments = TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::Host, Index >;
   constexpr int MaxValueSizeLog = HostSegments::MaxValueSizeLog();

   // Short segments together with a long one, so that blocks of all types are created
   const Index segmentsCount = 1000;
   TNL::Containers::Vector< Index, TNL::Devices::Host, Index > hostSizes( segmentsCount );
   hostSizes = 3;
   hostSizes[ 10 ] = 20000;
   hostSizes[ 500 ] = 300;

   HostSegments hostSegments( hostSizes );
   for( int i = 0; i < MaxValueSizeLog; i++ )
      EXPECT_EQ( hostSegments.getBlocks()[ i ].getSize(), 0 ) << "i = " << i;

#if defined( __CUDACC__ ) || defined( __HIP__ )
   using GPUSegments = TNL::Algorithms::Segments::AdaptiveCSR< TNL::Devices::GPU, Index >;

   // The segments assigned from the host must create the same blocks as the
   // segments set up directly on the GPU
   TNL::Containers::Vector< Index, TNL::Devices::GPU, Index > sizes( hostSizes );
   GPUSegments segments( sizes );
   GPUSegments assignedSegments;
   assignedSegments = hostSegments;
   for( int i = 0; i < MaxValueSizeLog; i++ ) {
      EXPECT_GT( segments.getBlocks()[ i ].getSize(), 0 ) << "i = " << i;
      EXPECT_EQ( assignedSegments.getBlocks()[ i ].getSize(), segments.getBlocks()[ i ].getSize() ) << "i = " << i;
   }

   // Count the elements of each segment using the adaptive kernels
   TNL::Containers::Vector< Index, TNL::Devices::GPU, Index > result( segmentsCount, 0 );
   auto result_view = result.getView();
   auto fetch = [] __cuda_callable__( Index segmentIdx, Index localIdx, Index globalIdx ) -> Index
   {
      return 1;
   };
   auto keep = [ = ] __cuda_callable__( Index i, Index a ) mutable
   {
      result_view[ i ] = a;
   };
   TNL::Algorithms::Segments::reduceAllSegments( assignedSegments, fetch, TNL::Plus{}, keep, static_cast< Index >( 0 ) );
   EXPECT_EQ( result, sizes );
#endif
}

TEST( AdaptiveCSRSegmentsTest, blocks_int )
{
   test_AdaptiveCSR_blocks< int >();
}

TEST( AdaptiveCSRSegmentsTest, blocks_long )
{
   test_AdaptiveCSR_blocks< long >();
}
