#pragma once

#include <limits>
#include <stdexcept>
#include <string>

#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Algorithms/Segments/traverse.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Functional.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Algorithms::Segments;

#if defined( __CUDACC__ )
using Device = Devices::Cuda;
#elif defined( __HIP__ )
using Device = Devices::Hip;
#else
using Device = Devices::Host;
#endif

constexpr bool isGPU = ! std::is_same_v< Device, Devices::Host >;

template< typename Segments >
Segments
makeSegments()
{
   using IndexType = typename Segments::IndexType;
   Containers::Vector< IndexType, Device, IndexType > segmentsSizes( 100 );
   segmentsSizes = 5;
   return Segments( segmentsSizes );
}

// Expects that all traversals of `segments` reject `launchConfig` with std::invalid_argument.
template< typename Segments >
void
expectTraversalRejects( const Segments& segments, const LaunchConfiguration& launchConfig )
{
   SCOPED_TRACE(
      "mapping " + std::to_string( static_cast< int >( launchConfig.getThreadsToSegmentsMapping() ) ) + ", "
      + std::to_string( launchConfig.getThreadsPerSegmentCount() ) + " TPS" );
   using IndexType = typename Segments::IndexType;
   Containers::Vector< IndexType, Device, IndexType > v( segments.getStorageSize(), 0 );
   auto v_view = v.getView();
   auto function =
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
   {
      v_view[ globalIdx ] = segmentIdx;
   };
   auto condition = [] __cuda_callable__( const IndexType segmentIdx ) -> bool
   {
      return segmentIdx % 2 == 0;
   };
   Containers::Vector< IndexType, Device, IndexType > segmentIndexes{ 1, 3, 5 };

   EXPECT_THROW( forAllElements( segments, function, launchConfig ), std::invalid_argument );
   EXPECT_THROW( forElements( segments, segmentIndexes, function, launchConfig ), std::invalid_argument );
   EXPECT_THROW( forAllElementsIf( segments, condition, function, launchConfig ), std::invalid_argument );
}

// Expects that all reductions of `segments` reject `launchConfig` with std::invalid_argument.
template< typename Segments >
void
expectReductionRejects( const Segments& segments, const LaunchConfiguration& launchConfig )
{
   SCOPED_TRACE(
      "mapping " + std::to_string( static_cast< int >( launchConfig.getThreadsToSegmentsMapping() ) ) + ", "
      + std::to_string( launchConfig.getThreadsPerSegmentCount() ) + " TPS" );
   using IndexType = typename Segments::IndexType;
   Containers::Vector< IndexType, Device, IndexType > result( segments.getSegmentCount() );
   auto result_view = result.getView();
   auto fetch = [] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> IndexType
   {
      return globalIdx;
   };
   auto reduction = [] __cuda_callable__( IndexType a, IndexType b ) -> IndexType
   {
      return a + b;
   };
   auto storer = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType value ) mutable
   {
      result_view[ segmentIdx ] = value;
   };
   auto storerWithIndexes =
      [ = ] __cuda_callable__( IndexType indexOfSegmentIdx, IndexType segmentIdx, IndexType value ) mutable
   {
      result_view[ segmentIdx ] = value;
   };
   auto storerWithArgument =
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType value, bool emptySegment ) mutable
   {
      result_view[ segmentIdx ] = value;
   };
   Containers::Vector< IndexType, Device, IndexType > segmentIndexes{ 1, 3, 5 };

   EXPECT_THROW( reduceAllSegments( segments, fetch, reduction, storer, (IndexType) 0, launchConfig ), std::invalid_argument );
   EXPECT_THROW(
      reduceSegments( segments, segmentIndexes, fetch, reduction, storerWithIndexes, (IndexType) 0, launchConfig ),
      std::invalid_argument );
   EXPECT_THROW(
      reduceAllSegmentsWithArgument( segments, fetch, TNL::MaxWithArg{}, storerWithArgument, launchConfig ),
      std::invalid_argument );
}

TEST( SegmentsLaunchConfigurationTest, threadsPerSegmentCountMustBePowerOfTwo )
{
   LaunchConfiguration launchConfig;
   for( int count : { 1, 2, 4, 32, 256, 1024 } ) {
      EXPECT_NO_THROW( launchConfig.setThreadsPerSegmentCount( count ) );
      EXPECT_EQ( launchConfig.getThreadsPerSegmentCount(), count );
   }
   for( int count : { 0, -1, -2, 3, 6, 12, 33, 100 } )
      EXPECT_THROW( launchConfig.setThreadsPerSegmentCount( count ), std::invalid_argument ) << "count = " << count;
   // a rejected count does not change the current one
   EXPECT_EQ( launchConfig.getThreadsPerSegmentCount(), 1024 );

   EXPECT_THROW( LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 3 ), std::invalid_argument );
   EXPECT_THROW( LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 0 ), std::invalid_argument );
}

// BiEllpack and ChunkedEllpack have a single native kernel for each operation,
// so they reject any other than the Default mapping on all devices.
template< typename Segments >
void
test_DefaultOnly()
{
   const Segments segments = makeSegments< Segments >();
   for( auto mapping : { ThreadsToSegmentsMapping::Uniform,
                         ThreadsToSegmentsMapping::PerElement,
                         ThreadsToSegmentsMapping::Tiered,
                         ThreadsToSegmentsMapping::Adaptive } )
   {
      SCOPED_TRACE( static_cast< int >( mapping ) );
      const LaunchConfiguration launchConfig( mapping, 1 );
      expectTraversalRejects( segments, launchConfig );
      expectReductionRejects( segments, launchConfig );
   }
}

TEST( SegmentsLaunchConfigurationTest, BiEllpackRejectsNonDefaultMapping )
{
   test_DefaultOnly< RowMajorBiEllpack< Device, int > >();
   test_DefaultOnly< ColumnMajorBiEllpack< Device, int > >();
}

TEST( SegmentsLaunchConfigurationTest, ChunkedEllpackRejectsNonDefaultMapping )
{
   test_DefaultOnly< RowMajorChunkedEllpack< Device, int > >();
   test_DefaultOnly< ColumnMajorChunkedEllpack< Device, int > >();
}

// The other formats resolve the launch configuration only on GPUs,
// the host kernels do not use it.
template< typename Segments >
void
test_CSR_UnsupportedLaunchConfigurations()
{
   const Segments segments = makeSegments< Segments >();
   // the CSR reduction has no PerElement kernel
   expectReductionRejects( segments, LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 1 ) );
   // the threads per segment counts above the supported ones
   expectReductionRejects( segments, LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 256 ) );
   expectTraversalRejects( segments, LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 128 ) );
   // a block of the Tiered kernels with 256 threads would process no segment
   expectReductionRejects( segments, LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 512 ) );
   expectTraversalRejects( segments, LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 512 ) );
}

TEST( SegmentsLaunchConfigurationTest, CSRRejectsUnsupportedLaunchConfigurations )
{
   if constexpr( isGPU )
      test_CSR_UnsupportedLaunchConfigurations< CSR< Device, int > >();
}

template< typename Segments >
void
test_Ellpack_UnsupportedLaunchConfigurations()
{
   const Segments segments = makeSegments< Segments >();
   // the reductions of Ellpack formats support only the Uniform mapping
   for( auto mapping : { ThreadsToSegmentsMapping::PerElement, ThreadsToSegmentsMapping::Tiered } ) {
      SCOPED_TRACE( static_cast< int >( mapping ) );
      expectReductionRejects( segments, LaunchConfiguration( mapping, 1 ) );
   }
   // the traversals of Ellpack formats support only the Uniform and PerElement mappings
   for( auto mapping : { ThreadsToSegmentsMapping::Tiered, ThreadsToSegmentsMapping::Adaptive } ) {
      SCOPED_TRACE( static_cast< int >( mapping ) );
      expectTraversalRejects( segments, LaunchConfiguration( mapping, 1 ) );
   }
   expectTraversalRejects( segments, LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 128 ) );
}

TEST( SegmentsLaunchConfigurationTest, EllpackRejectsUnsupportedLaunchConfigurations )
{
   if constexpr( isGPU ) {
      test_Ellpack_UnsupportedLaunchConfigurations< RowMajorEllpack< Device, int > >();
      test_Ellpack_UnsupportedLaunchConfigurations< ColumnMajorEllpack< Device, int > >();
      // the row-major reduction supports at most 64 threads per segment
      expectReductionRejects(
         makeSegments< RowMajorEllpack< Device, int > >(), LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 128 ) );
   }
}

template< typename Segments >
void
test_SlicedEllpack_UnsupportedThreadsPerSegmentCounts()
{
   using IndexType = typename Segments::IndexType;
   const Segments segments = makeSegments< Segments >();
   // the row-major reduction supports at most 64 threads per segment
   // (the reductions with segment indexes or with an argument have a single kernel which ignores the count)
   EXPECT_THROW(
      reduceAllSegments(
         segments,
         [] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> IndexType
         {
            return globalIdx;
         },
         TNL::Plus{},
         [] __cuda_callable__( IndexType segmentIdx, IndexType value ) {},
         LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 128 ) ),
      std::invalid_argument );
   // a block of the PerElement traversal of a range of segments must process whole slices,
   // i.e. 256 / threadsPerSegmentCount must be a multiple of the slice size
   // (the traversals with segment indexes or with a condition process single segments)
   static_assert( Segments::getSliceSize() > 256 / 64 );
   EXPECT_THROW(
      forAllElements(
         segments,
         [] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) {},
         LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 64 ) ),
      std::invalid_argument );
}

TEST( SegmentsLaunchConfigurationTest, SlicedEllpackRejectsUnsupportedLaunchConfigurations )
{
   if constexpr( isGPU ) {
      test_Ellpack_UnsupportedLaunchConfigurations< RowMajorSlicedEllpack< Device, int > >();
      test_Ellpack_UnsupportedLaunchConfigurations< ColumnMajorSlicedEllpack< Device, int > >();
      test_SlicedEllpack_UnsupportedThreadsPerSegmentCounts< RowMajorSlicedEllpack< Device, int > >();
   }
}

#include "../../main.h"
