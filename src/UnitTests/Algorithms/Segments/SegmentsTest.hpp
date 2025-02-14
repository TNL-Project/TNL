#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/Segments/traverse.h>
#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Algorithms/Segments/find.h>
#include <TNL/Math.h>

#include <iostream>
#include <gtest/gtest.h>

template< typename Segments >
void
test_SetSegmentsSizes_EqualSizes()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   // Test default-initialized segment acts the same as an empty segment
   Segments defaultSegments;

   EXPECT_EQ( defaultSegments.getSegmentsCount(), 0 );
   EXPECT_EQ( defaultSegments.getSize(), 0 );
   EXPECT_LE( defaultSegments.getSize(), defaultSegments.getStorageSize() );

   // Test with one segment
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > oneSegmentsSizes( 1 );
   oneSegmentsSizes = 1;

   Segments oneSegments( oneSegmentsSizes );

   EXPECT_EQ( oneSegments.getSegmentsCount(), 1 );
   EXPECT_EQ( oneSegments.getSize(), 1 );
   EXPECT_LE( oneSegments.getSize(), oneSegments.getStorageSize() );

   // Test setup with empty segments
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > emptySegmentsSizes( 0 );
   emptySegmentsSizes = 0;

   Segments emptySegments( emptySegmentsSizes );

   EXPECT_EQ( emptySegments.getSegmentsCount(), 0 );
   EXPECT_EQ( emptySegments.getSize(), 0 );
   EXPECT_LE( emptySegments.getSize(), emptySegments.getStorageSize() );

   const IndexType segmentsCount = 20;
   const IndexType segmentSize = 5;
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes = segmentSize;

   Segments segments( segmentsSizes );

   EXPECT_EQ( segments.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segments.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segments.getSize(), segments.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      // Some formats may use padding zeros and allocate more slots than the segment size
      EXPECT_GE( segments.getSegmentSize( i ), segmentSize );

   Segments segments2( segments );
   EXPECT_EQ( segments2.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segments2.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segments2.getSize(), segments2.getStorageSize() );
   for( IndexType i = 0; i < segmentsCount; i++ )
      // Some formats may use padding zeros and allocate more slots than the segment size
      EXPECT_GE( segments2.getSegmentSize( i ), segmentSize );

   Segments segments3;
   segments3.setSegmentsSizes( segmentsSizes );

   EXPECT_EQ( segments3.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segments3.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segments3.getSize(), segments3.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      // Some formats may use padding zeros and allocate more slots than the segment size
      EXPECT_GE( segments3.getSegmentSize( i ), segmentSize );

   using SegmentsView = typename Segments::ViewType;

   SegmentsView segmentsView = segments.getView();
   EXPECT_EQ( segmentsView.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segmentsView.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segmentsView.getSize(), segments.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      // Some formats may use padding zeros and allocate more slots than the segment size
      EXPECT_GE( segmentsView.getSegmentSize( i ), segmentSize );
}

template< typename Segments >
void
test_SetSegmentsSizes_EqualSizes_EllpackOnly()
{
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 20;
   const IndexType segmentSize = 5;

   Segments segments( segmentsCount, segmentSize );

   EXPECT_EQ( segments.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segments.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segments.getSize(), segments.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segments.getSegmentSize( i ), segmentSize );

   Segments segments2( segments );
   EXPECT_EQ( segments2.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segments2.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segments2.getSize(), segments2.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segments2.getSegmentSize( i ), segmentSize );

   Segments segments3;
   segments3.setSegmentsSizes( segmentsCount, segmentSize );

   EXPECT_EQ( segments3.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segments3.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segments3.getSize(), segments3.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segments3.getSegmentSize( i ), segmentSize );

   using SegmentsView = typename Segments::ViewType;

   SegmentsView segmentsView = segments.getView();
   EXPECT_EQ( segmentsView.getSegmentsCount(), segmentsCount );
   EXPECT_EQ( segmentsView.getSize(), segmentsCount * segmentSize );
   EXPECT_LE( segmentsView.getSize(), segments.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segmentsView.getSegmentSize( i ), segmentSize );
}

template< typename Segments >
void
test_findInSegments()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 10;
   const IndexType maxSegmentSize = 7;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx % maxSegmentSize + 1;
      } );

   Segments segments( segmentsSizes );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize() );
   v = -1;

   auto view = v.getView();
   auto init =
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable -> bool
   {
      TNL_ASSERT_LT( globalIdx, view.getSize(), "" );
      // some segments may use padding zeros and their size may be greater than the original segment size
      if( localIdx <= segmentIdx % maxSegmentSize )
         view[ globalIdx ] = localIdx + 1;
      return true;
   };
   TNL::Algorithms::Segments::forAllElements( segments, init );

   TNL::Containers::Vector< bool, DeviceType, IndexType > found( segmentsCount, false );
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > positions( segmentsCount, -1 );

   const auto v_view = v.getConstView();
   auto found_view = found.getView();
   auto positions_view = positions.getView();
   auto condition = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> bool
   {
      return v_view[ globalIdx ] == 5;
   };
   auto keep = [ = ] __cuda_callable__( const IndexType segmentIdx, bool found, const IndexType localIdx ) mutable
   {
      found_view[ segmentIdx ] = found;
      if( found )
         positions_view[ segmentIdx ] = localIdx;
      else
         positions_view[ segmentIdx ] = -1;
   };
   TNL::Algorithms::Segments::findInAllSegments( segments, condition, keep );

   for( IndexType i = 0; i < segmentsCount; i++ ) {
      if( i % 7 >= 4 ) {
         EXPECT_EQ( found.getElement( i ), true ) << "segmentIdx = " << i;
         EXPECT_EQ( positions.getElement( i ), 4 ) << "segmentIdx = " << i;
      }
      else {
         EXPECT_EQ( found.getElement( i ), false ) << "segmentIdx = " << i;
         EXPECT_EQ( positions.getElement( i ), -1 ) << "segmentIdx = " << i;
      }
   }

   // Test with segments view and short fetch
   found_view = false;
   positions_view = -1;
   auto short_condition = [ = ] __cuda_callable__( IndexType globalIdx ) -> bool
   {
      return v_view[ globalIdx ] == 5;
   };

   TNL::Algorithms::Segments::findInAllSegments( segments.getView(), short_condition, keep );

   for( IndexType i = 0; i < segmentsCount; i++ ) {
      if( i % 7 >= 4 ) {
         EXPECT_EQ( found.getElement( i ), true ) << "segmentIdx = " << i;
         EXPECT_EQ( positions.getElement( i ), 4 ) << "segmentIdx = " << i;
      }
      else {
         EXPECT_EQ( found.getElement( i ), false ) << "segmentIdx = " << i;
         EXPECT_EQ( positions.getElement( i ), -1 ) << "segmentIdx = " << i;
      }
   }
}

template< typename Segments >
void
test_findInSegmentsWithIndexes()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 10;
   const IndexType maxSegmentSize = 7;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx % maxSegmentSize + 1;
      } );

   Segments segments( segmentsSizes );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize() ),
      segmentIndexes( segmentsCount / 2 );
   segmentIndexes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx * 2;
      } );
   v = -1;

   auto view = v.getView();
   auto init =
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable -> bool
   {
      TNL_ASSERT_LT( globalIdx, view.getSize(), "" );
      // some segments may use padding zeros and their size may be greater than the original segment size
      if( localIdx <= segmentIdx % maxSegmentSize )
         view[ globalIdx ] = localIdx + 1;
      return true;
   };
   TNL::Algorithms::Segments::forAllElements( segments, init );

   TNL::Containers::Vector< bool, DeviceType, IndexType > found( segmentsCount, false );
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > positions( segmentsCount, -1 );

   const auto v_view = v.getConstView();
   auto found_view = found.getView();
   auto positions_view = positions.getView();
   auto condition = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> bool
   {
      return v_view[ globalIdx ] == 5;
   };
   auto keep = [ = ] __cuda_callable__(
                  const IndexType segmentIdx_idx, const IndexType segmentIdx, bool found, const IndexType localIdx ) mutable
   {
      found_view[ segmentIdx ] = found;
      if( found )
         positions_view[ segmentIdx ] = localIdx;
      else
         positions_view[ segmentIdx ] = -1;
   };
   TNL::Algorithms::Segments::findInSegments( segments, segmentIndexes, condition, keep );

   for( IndexType i = 0; i < segmentsCount; i++ ) {
      if( i % 7 >= 4 && i % 2 == 0 ) {
         EXPECT_EQ( found.getElement( i ), true ) << "segmentIdx = " << i;
         EXPECT_EQ( positions.getElement( i ), 4 ) << "segmentIdx = " << i;
      }
      else {
         EXPECT_EQ( found.getElement( i ), false ) << "segmentIdx = " << i;
         EXPECT_EQ( positions.getElement( i ), -1 ) << "segmentIdx = " << i;
      }
   }

   // Test with segments view and short fetch
   found_view = false;
   positions_view = -1;
   auto short_condition = [ = ] __cuda_callable__( IndexType globalIdx ) -> bool
   {
      return v_view[ globalIdx ] == 5;
   };

   TNL::Algorithms::Segments::findInSegments( segments.getView(), segmentIndexes, short_condition, keep );

   for( IndexType i = 0; i < segmentsCount; i++ ) {
      if( i % 7 >= 4 && i % 2 == 0 ) {
         EXPECT_EQ( found.getElement( i ), true ) << "segmentIdx = " << i;
         EXPECT_EQ( positions.getElement( i ), 4 ) << "segmentIdx = " << i;
      }
      else {
         EXPECT_EQ( found.getElement( i ), false ) << "segmentIdx = " << i;
         EXPECT_EQ( positions.getElement( i ), -1 ) << "segmentIdx = " << i;
      }
   }
}

template< typename Segments >
void
test_findInSegmentsIf()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 10;
   const IndexType maxSegmentSize = 7;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx % maxSegmentSize + 1;
      } );

   Segments segments( segmentsSizes );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize() );
   v = -1;

   auto view = v.getView();
   auto init =
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable -> bool
   {
      TNL_ASSERT_LT( globalIdx, view.getSize(), "" );
      // some segments may use padding zeros and their size may be greater than the original segment size
      if( localIdx <= segmentIdx % maxSegmentSize )
         view[ globalIdx ] = localIdx + 1;
      return true;
   };
   TNL::Algorithms::Segments::forAllElements( segments, init );

   TNL::Containers::Vector< bool, DeviceType, IndexType > found( segmentsCount, false );
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > positions( segmentsCount, -1 );

   const auto v_view = v.getConstView();
   auto found_view = found.getView();
   auto positions_view = positions.getView();
   auto segmentCondition = [ = ] __cuda_callable__( IndexType segmentIdx ) -> bool
   {
      return segmentIdx % 2 == 0;
   };
   auto condition = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> bool
   {
      return v_view[ globalIdx ] == 5;
   };
   auto keep = [ = ] __cuda_callable__( const IndexType segmentIdx, bool found, const IndexType localIdx ) mutable
   {
      found_view[ segmentIdx ] = found;
      if( found )
         positions_view[ segmentIdx ] = localIdx;
      else
         positions_view[ segmentIdx ] = -1;
   };
   TNL::Algorithms::Segments::findInSegmentsIf( segments, segmentCondition, condition, keep );

   for( IndexType i = 0; i < segmentsCount; i++ ) {
      if( i % 7 >= 4 && i % 2 == 0 ) {
         EXPECT_EQ( found.getElement( i ), true ) << "segmentIdx = " << i;
         EXPECT_EQ( positions.getElement( i ), 4 ) << "segmentIdx = " << i;
      }
      else {
         EXPECT_EQ( found.getElement( i ), false ) << "segmentIdx = " << i;
         EXPECT_EQ( positions.getElement( i ), -1 ) << "segmentIdx = " << i;
      }
   }

   // Test with segments view and short fetch
   found_view = false;
   positions_view = -1;
   auto short_condition = [ = ] __cuda_callable__( IndexType globalIdx ) -> bool
   {
      return v_view[ globalIdx ] == 5;
   };

   TNL::Algorithms::Segments::findInSegmentsIf( segments.getView(), segmentCondition, short_condition, keep );

   for( IndexType i = 0; i < segmentsCount; i++ ) {
      if( i % 7 >= 4 && i % 2 == 0 ) {
         EXPECT_EQ( found.getElement( i ), true ) << "segmentIdx = " << i;
         EXPECT_EQ( positions.getElement( i ), 4 ) << "segmentIdx = " << i;
      }
      else {
         EXPECT_EQ( found.getElement( i ), false ) << "segmentIdx = " << i;
         EXPECT_EQ( positions.getElement( i ), -1 ) << "segmentIdx = " << i;
      }
   }
}
