#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/Segments/TypeTraits.h>
#include <TNL/Algorithms/Segments/traverse.h>
#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Algorithms/Segments/find.h>
#include <TNL/Algorithms/Segments/sort.h>
#include <TNL/Algorithms/Segments/scan.h>
#include <TNL/Math.h>

#include <iostream>
#include <gtest/gtest.h>

template< typename Segments, bool isSorted = TNL::Algorithms::Segments::isSortedSegments_v< Segments > >
struct HostSegmentsGetter
{
   using type = typename Segments::template Self< TNL::Devices::Host >;
};

template< typename Segments >
struct HostSegmentsGetter< Segments, true >
{
   using type =
      TNL::Algorithms::Segments::SortedSegments< typename Segments::EmbeddedSegmentsType::template Self< TNL::Devices::Host > >;
};

template< typename Segments >
void
test_isSegments()
{
   EXPECT_EQ( TNL::Algorithms::Segments::isSegments_v< Segments >, true );
}

template< typename Segments >
void
test_getView()
{
   Segments segments;
   EXPECT_EQ( segments.getSegmentCount(), 0 );

   auto view = segments.getView();
   EXPECT_EQ( view.getSegmentCount(), 0 );

   auto const_view = segments.getConstView();
   EXPECT_EQ( const_view.getSegmentCount(), 0 );
}

template< typename Segments >
void
test_setSegmentsSizes_EqualSizes()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   // Test default-initialized segment acts the same as an empty segment
   Segments defaultSegments;

   EXPECT_EQ( defaultSegments.getSegmentCount(), 0 );
   EXPECT_EQ( defaultSegments.getElementCount(), 0 );
   EXPECT_LE( defaultSegments.getElementCount(), defaultSegments.getStorageSize() );

   // Test with one segment
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > oneSegmentsSizes( 1 );
   oneSegmentsSizes = 1;

   Segments oneSegments( oneSegmentsSizes );

   EXPECT_EQ( oneSegments.getSegmentCount(), 1 );
   EXPECT_EQ( oneSegments.getElementCount(), 1 );
   EXPECT_LE( oneSegments.getElementCount(), oneSegments.getStorageSize() );

   // Test setup with empty segments
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > emptySegmentsSizes( 0 );
   emptySegmentsSizes = 0;

   Segments emptySegments( emptySegmentsSizes );

   EXPECT_EQ( emptySegments.getSegmentCount(), 0 );
   EXPECT_EQ( emptySegments.getElementCount(), 0 );
   EXPECT_LE( emptySegments.getElementCount(), emptySegments.getStorageSize() );

   const IndexType segmentsCount = 20;
   const IndexType segmentSize = 5;
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes = segmentSize;

   Segments segments( segmentsSizes );

   EXPECT_EQ( segments.getSegmentCount(), segmentsCount );
   EXPECT_EQ( segments.getElementCount(), segmentsCount * segmentSize );
   EXPECT_LE( segments.getElementCount(), segments.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      // Some formats may use padding zeros and allocate more slots than the segment size
      EXPECT_GE( segments.getSegmentSize( i ), segmentSize );

   Segments segments2( segments );
   EXPECT_EQ( segments2.getSegmentCount(), segmentsCount );
   EXPECT_EQ( segments2.getElementCount(), segmentsCount * segmentSize );
   EXPECT_LE( segments2.getElementCount(), segments2.getStorageSize() );
   for( IndexType i = 0; i < segmentsCount; i++ )
      // Some formats may use padding zeros and allocate more slots than the segment size
      EXPECT_GE( segments2.getSegmentSize( i ), segmentSize );

   Segments segments3;
   segments3.setSegmentsSizes( segmentsSizes );

   EXPECT_EQ( segments3.getSegmentCount(), segmentsCount );
   EXPECT_EQ( segments3.getElementCount(), segmentsCount * segmentSize );
   EXPECT_LE( segments3.getElementCount(), segments3.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      // Some formats may use padding zeros and allocate more slots than the segment size
      EXPECT_GE( segments3.getSegmentSize( i ), segmentSize );

   using SegmentsView = typename Segments::ViewType;

   SegmentsView segmentsView = segments.getView();
   EXPECT_EQ( segmentsView.getSegmentCount(), segmentsCount );
   EXPECT_EQ( segmentsView.getElementCount(), segmentsCount * segmentSize );
   EXPECT_LE( segmentsView.getElementCount(), segments.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      // Some formats may use padding zeros and allocate more slots than the segment size
      EXPECT_GE( segmentsView.getSegmentSize( i ), segmentSize );
}

template< typename Segments >
void
test_setSegmentsSizes_EqualSizes_EllpackOnly()
{
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 20;
   const IndexType segmentSize = 5;

   Segments segments( segmentsCount, segmentSize );

   EXPECT_EQ( segments.getSegmentCount(), segmentsCount );
   EXPECT_EQ( segments.getElementCount(), segmentsCount * segmentSize );
   EXPECT_LE( segments.getElementCount(), segments.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segments.getSegmentSize( i ), segmentSize );

   Segments segments2( segments );
   EXPECT_EQ( segments2.getSegmentCount(), segmentsCount );
   EXPECT_EQ( segments2.getElementCount(), segmentsCount * segmentSize );
   EXPECT_LE( segments2.getElementCount(), segments2.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segments2.getSegmentSize( i ), segmentSize );

   Segments segments3;
   segments3.setSegmentsSizes( segmentsCount, segmentSize );

   EXPECT_EQ( segments3.getSegmentCount(), segmentsCount );
   EXPECT_EQ( segments3.getElementCount(), segmentsCount * segmentSize );
   EXPECT_LE( segments3.getElementCount(), segments3.getStorageSize() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( segments3.getSegmentSize( i ), segmentSize );

   using SegmentsView = typename Segments::ViewType;

   SegmentsView segmentsView = segments.getView();
   EXPECT_EQ( segmentsView.getSegmentCount(), segmentsCount );
   EXPECT_EQ( segmentsView.getElementCount(), segmentsCount * segmentSize );
   EXPECT_LE( segmentsView.getElementCount(), segments.getStorageSize() );

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
   auto keep = [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, bool found ) mutable
   {
      found_view[ segmentIdx ] = found;
      if( found )
         positions_view[ segmentIdx ] = localIdx;
      else
         positions_view[ segmentIdx ] = -1;
   };
   TNL::Algorithms::Segments::findInAllSegments( segments, condition, keep );

   for( IndexType i = 0; i < segmentsCount; i++ ) {
      if( i % maxSegmentSize >= 4 ) {
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
      if( i % maxSegmentSize >= 4 ) {
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
                  const IndexType segmentIdx_idx, const IndexType segmentIdx, const IndexType localIdx, bool found ) mutable
   {
      found_view[ segmentIdx ] = found;
      if( found )
         positions_view[ segmentIdx ] = localIdx;
      else
         positions_view[ segmentIdx ] = -1;
   };
   TNL::Algorithms::Segments::findInSegments( segments, segmentIndexes, condition, keep );

   for( IndexType i = 0; i < segmentsCount; i++ ) {
      if( i % maxSegmentSize >= 4 && i % 2 == 0 ) {
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
      if( i % maxSegmentSize >= 4 && i % 2 == 0 ) {
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
   auto keep = [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, bool found ) mutable
   {
      found_view[ segmentIdx ] = found;
      if( found )
         positions_view[ segmentIdx ] = localIdx;
      else
         positions_view[ segmentIdx ] = -1;
   };
   TNL::Algorithms::Segments::findInAllSegmentsIf( segments, segmentCondition, condition, keep );

   for( IndexType i = 0; i < segmentsCount; i++ ) {
      if( i % maxSegmentSize >= 4 && i % 2 == 0 ) {
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

   TNL::Algorithms::Segments::findInAllSegmentsIf( segments.getView(), segmentCondition, short_condition, keep );

   for( IndexType i = 0; i < segmentsCount; i++ ) {
      if( i % maxSegmentSize >= 4 && i % 2 == 0 ) {
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
test_sortSegments()
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

   auto v_view = v.getView();
   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
      {
         TNL_ASSERT_LT( globalIdx, v_view.getSize(), "" );
         // some segments may use padding zeros and their size may be greater than the original segment size
         if( localIdx <= segmentIdx % maxSegmentSize )
            v_view[ globalIdx ] = localIdx + 1;
         else
            v_view[ globalIdx ] = std::numeric_limits< IndexType >::lowest();
      } );

   auto fetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> IndexType
   {
      return v_view[ globalIdx ];
   };
   auto descending_compare = [ = ] __cuda_callable__( const IndexType a, const IndexType b ) -> bool
   {
      return a >= b;
   };
   auto ascending_compare = [ = ] __cuda_callable__( const IndexType a, const IndexType b ) -> bool
   {
      return a <= b;
   };
   auto swap_lambda = [ = ] __cuda_callable__( const IndexType globalIdx1, const IndexType globalIdx2 ) mutable
   {
      TNL::swap( v_view[ globalIdx1 ], v_view[ globalIdx2 ] );
   };
   TNL::Algorithms::Segments::sortAllSegments( segments, fetch, descending_compare, swap_lambda );

   for( IndexType i = 0; i < segmentsCount; i++ ) {
      for( IndexType j = 0; j < i % maxSegmentSize; j++ ) {
         EXPECT_GE( v.getElement( segments.getGlobalIndex( i, j ) ), v.getElement( segments.getGlobalIndex( i, j + 1 ) ) )
            << "segmentIdx = " << i << " localIdx = " << j;
      }
   }

   // Test with segments view and short fetch
   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
      {
         TNL_ASSERT_LT( globalIdx, v_view.getSize(), "" );
         // some segments may use padding zeros and their size may be greater than the original segment size
         if( localIdx > segmentIdx % maxSegmentSize )
            v_view[ globalIdx ] = std::numeric_limits< IndexType >::max();
      } );
   auto short_fetch = [ = ] __cuda_callable__( IndexType globalIdx ) -> IndexType
   {
      return v_view[ globalIdx ];
   };

   TNL::Algorithms::Segments::sortAllSegments( segments.getView(), short_fetch, ascending_compare, swap_lambda );

   for( IndexType i = 0; i < segmentsCount; i++ ) {
      for( IndexType j = 0; j < i % maxSegmentSize; j++ ) {
         EXPECT_LE( v.getElement( segments.getGlobalIndex( i, j ) ), v.getElement( segments.getGlobalIndex( i, j + 1 ) ) )
            << "segmentIdx = " << i << " localIdx = " << j;
      }
   }
}

template< typename Segments >
void
test_sortSegmentsWithSegmentIndexes()
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
   v = -1;
   segmentIndexes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx * 2;
      } );

   auto v_view = v.getView();
   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
      {
         TNL_ASSERT_LT( globalIdx, v_view.getSize(), "" );
         // some segments may use padding zeros and their size may be greater than the original segment size
         if( localIdx <= segmentIdx % maxSegmentSize )
            v_view[ globalIdx ] = localIdx + 1;
         else
            v_view[ globalIdx ] = std::numeric_limits< IndexType >::lowest();
      } );

   auto fetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> IndexType
   {
      return v_view[ globalIdx ];
   };
   auto descending_compare = [ = ] __cuda_callable__( const IndexType a, const IndexType b ) -> bool
   {
      return a >= b;
   };
   auto ascending_compare = [ = ] __cuda_callable__( const IndexType a, const IndexType b ) -> bool
   {
      return a <= b;
   };
   auto swap_lambda = [ = ] __cuda_callable__( const IndexType globalIdx1, const IndexType globalIdx2 ) mutable
   {
      TNL::swap( v_view[ globalIdx1 ], v_view[ globalIdx2 ] );
   };
   TNL::Algorithms::Segments::sortSegments( segments, segmentIndexes, fetch, descending_compare, swap_lambda );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentIdx % maxSegmentSize; localIdx++ ) {
         if( segmentIdx % 2 == 0 )
            EXPECT_GE( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ),
                       v.getElement( segments.getGlobalIndex( segmentIdx, localIdx + 1 ) ) )
               << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx;
         else
            EXPECT_LE( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ),
                       v.getElement( segments.getGlobalIndex( segmentIdx, localIdx + 1 ) ) )
               << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx;
      }
   }

   // Test with segments view and short fetch
   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
      {
         TNL_ASSERT_LT( globalIdx, v_view.getSize(), "" );
         // some segments may use padding zeros and their size may be greater than the original segment size
         if( localIdx > segmentIdx % maxSegmentSize )
            v_view[ globalIdx ] = std::numeric_limits< IndexType >::max();
      } );
   auto short_fetch = [ = ] __cuda_callable__( IndexType globalIdx ) -> IndexType
   {
      return v_view[ globalIdx ];
   };

   TNL::Algorithms::Segments::sortSegments( segments.getView(), segmentIndexes, short_fetch, ascending_compare, swap_lambda );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentIdx % maxSegmentSize; localIdx++ ) {
         // All segments with are now sorted in descending order
         EXPECT_LE( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ),
                    v.getElement( segments.getGlobalIndex( segmentIdx, localIdx + 1 ) ) )
            << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx;
      }
   }
}

template< typename Segments >
void
test_sortSegmentsIf()
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

   auto v_view = v.getView();
   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
      {
         TNL_ASSERT_LT( globalIdx, v_view.getSize(), "" );
         // some segments may use padding zeros and their size may be greater than the original segment size
         if( localIdx <= segmentIdx % maxSegmentSize )
            v_view[ globalIdx ] = localIdx + 1;
         else
            v_view[ globalIdx ] = std::numeric_limits< IndexType >::lowest();
      } );

   auto condition = [ = ] __cuda_callable__( IndexType segmentIdx ) -> bool
   {
      return segmentIdx % 2 == 0;
   };
   auto fetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> IndexType
   {
      return v_view[ globalIdx ];
   };
   auto descending_compare = [ = ] __cuda_callable__( const IndexType a, const IndexType b ) -> bool
   {
      return a >= b;
   };
   auto ascending_compare = [ = ] __cuda_callable__( const IndexType a, const IndexType b ) -> bool
   {
      return a <= b;
   };
   auto swap_lambda = [ = ] __cuda_callable__( const IndexType globalIdx1, const IndexType globalIdx2 ) mutable
   {
      TNL::swap( v_view[ globalIdx1 ], v_view[ globalIdx2 ] );
   };
   TNL::Algorithms::Segments::sortAllSegmentsIf( segments, condition, fetch, descending_compare, swap_lambda );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentIdx % maxSegmentSize; localIdx++ ) {
         if( segmentIdx % 2 == 0 )
            EXPECT_GE( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ),
                       v.getElement( segments.getGlobalIndex( segmentIdx, localIdx + 1 ) ) )
               << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx;
         else
            EXPECT_LE( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ),
                       v.getElement( segments.getGlobalIndex( segmentIdx, localIdx + 1 ) ) )
               << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx;
      }
   }

   // Test with segments view and short fetch
   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
      {
         TNL_ASSERT_LT( globalIdx, v_view.getSize(), "" );
         // some segments may use padding zeros and their size may be greater than the original segment size
         if( localIdx > segmentIdx % maxSegmentSize )
            v_view[ globalIdx ] = std::numeric_limits< IndexType >::max();
      } );
   auto short_fetch = [ = ] __cuda_callable__( IndexType globalIdx ) -> IndexType
   {
      return v_view[ globalIdx ];
   };

   TNL::Algorithms::Segments::sortAllSegmentsIf( segments.getView(), condition, short_fetch, ascending_compare, swap_lambda );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentIdx % maxSegmentSize; localIdx++ ) {
         // All segments with are now sorted in descending order
         EXPECT_LE( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ),
                    v.getElement( segments.getGlobalIndex( segmentIdx, localIdx + 1 ) ) )
            << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx;
      }
   }
}

template< typename Segments >
void
test_scanSegments()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;
   using ValueType = double;
   using HostSegments = typename HostSegmentsGetter< Segments >::type;

   // Setup segments with varying sizes
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes{ 1, 2, 3, 4, 5 };
   Segments segments( segmentsSizes );

   // Setup test data
   TNL::Containers::Vector< ValueType, DeviceType > inclusive_result( segments.getStorageSize() ),
      exclusive_result( segments.getStorageSize() ), data( segments.getStorageSize() );
   auto data_view = data.getView();
   auto inclusive_result_view = inclusive_result.getView();
   auto exclusive_result_view = exclusive_result.getView();
   auto segmentsSizes_view = segmentsSizes.getConstView();
   data_view = 0.0;
   exclusive_result = 0.0;
   inclusive_result = 0.0;

   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < segmentsSizes_view[ segmentIdx ] ) {
            data_view[ globalIdx ] = segmentIdx + 1.0;
            inclusive_result_view[ globalIdx ] = ( segmentIdx + 1.0 ) * ( localIdx + 1.0 );
            exclusive_result_view[ globalIdx ] = inclusive_result_view[ globalIdx ] - data_view[ globalIdx ];
         }
      } );

   // Test inclusive scan
   auto fetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> ValueType
   {
      if( localIdx < segmentsSizes_view[ segmentIdx ] )
         return data_view[ globalIdx ];
      else
         return 0.0;
   };
   auto write = [ = ] __cuda_callable__( IndexType globalIdx, ValueType value ) mutable
   {
      data_view[ globalIdx ] = value;
   };

   TNL::Algorithms::Segments::inclusiveScanAllSegments( segments, fetch, TNL::Plus{}, write );

   HostSegments hostSegments;
   hostSegments = segments;
   TNL::Algorithms::Segments::forAllElements(
      hostSegments,
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < segmentsSizes_view.getElement( segmentIdx ) ) {
            EXPECT_EQ( data_view.getElement( globalIdx ), inclusive_result_view.getElement( globalIdx ) )
               << "segmentIdx = " << segmentIdx << ", localIdx = " << localIdx
               << " segmentSize = " << segmentsSizes_view.getElement( segmentIdx );
         }
      } );

   // Reset data for exclusive scan test
   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < segmentsSizes_view[ segmentIdx ] ) {
            data_view[ globalIdx ] = segmentIdx + 1.0;
         }
      } );
   TNL::Algorithms::Segments::exclusiveScanAllSegments( segments, fetch, TNL::Plus{}, write );

   TNL::Algorithms::Segments::forAllElements(
      hostSegments,
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < segmentsSizes_view.getElement( segmentIdx ) ) {
            EXPECT_EQ( data_view.getElement( globalIdx ), exclusive_result_view.getElement( globalIdx ) )
               << "segmentIdx = " << segmentIdx << ", localIdx = " << localIdx;
         }
      } );
}

template< typename Segments >
void
test_scanSegmentsWithSegmentIndexes()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;
   using ValueType = double;
   using HostSegments = typename HostSegmentsGetter< Segments >::type;

   // Setup segments with varying sizes
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes{ 1, 2, 3, 4, 5 }, segmentIndexes{ 0, 2, 4 };
   Segments segments( segmentsSizes );

   // Setup test data
   TNL::Containers::Vector< ValueType, DeviceType > inclusive_result( segments.getStorageSize() ),
      exclusive_result( segments.getStorageSize() ), data( segments.getStorageSize() );
   auto data_view = data.getView();
   auto inclusive_result_view = inclusive_result.getView();
   auto exclusive_result_view = exclusive_result.getView();
   auto segmentsSizes_view = segmentsSizes.getConstView();
   data_view = 0.0;
   exclusive_result = 0.0;
   inclusive_result = 0.0;

   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < segmentsSizes_view[ segmentIdx ] ) {
            data_view[ globalIdx ] = segmentIdx + 1.0;
            if( segmentIdx % 2 == 0 ) {
               inclusive_result_view[ globalIdx ] = ( segmentIdx + 1.0 ) * ( localIdx + 1.0 );
               exclusive_result_view[ globalIdx ] = inclusive_result_view[ globalIdx ] - data_view[ globalIdx ];
            }
            else {
               inclusive_result_view[ globalIdx ] = data_view[ globalIdx ];
               exclusive_result_view[ globalIdx ] = data_view[ globalIdx ];
            }
         }
      } );

   // Test inclusive scan
   auto fetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> ValueType
   {
      if( localIdx < segmentsSizes_view[ segmentIdx ] )
         return data_view[ globalIdx ];
      else
         return 0.0;
   };
   auto write = [ = ] __cuda_callable__( IndexType globalIdx, ValueType value ) mutable
   {
      data_view[ globalIdx ] = value;
   };

   TNL::Algorithms::Segments::inclusiveScanSegments( segments, segmentIndexes, fetch, TNL::Plus{}, write );

   HostSegments hostSegments;
   hostSegments = segments;
   TNL::Algorithms::Segments::forAllElements(
      hostSegments,
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < segmentsSizes_view.getElement( segmentIdx ) ) {
            EXPECT_EQ( data_view.getElement( globalIdx ), inclusive_result_view.getElement( globalIdx ) )
               << "segmentIdx = " << segmentIdx << ", localIdx = " << localIdx;
         }
      } );

   // Reset data for exclusive scan test
   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < segmentsSizes_view[ segmentIdx ] ) {
            data_view[ globalIdx ] = segmentIdx + 1.0;
         }
      } );

   TNL::Algorithms::Segments::exclusiveScanSegments( segments, segmentIndexes, fetch, TNL::Plus{}, write );

   TNL::Algorithms::Segments::forAllElements(
      hostSegments,
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < segmentsSizes_view.getElement( segmentIdx ) ) {
            EXPECT_EQ( data_view.getElement( globalIdx ), exclusive_result_view.getElement( globalIdx ) )
               << "segmentIdx = " << segmentIdx << ", localIdx = " << localIdx;
         }
      } );
}

template< typename Segments >
void
test_scanSegmentsIf()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;
   using ValueType = double;
   using HostSegments = typename HostSegmentsGetter< Segments >::type;

   // Setup segments with varying sizes
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes{ 1, 2, 3, 4, 5 };
   Segments segments( segmentsSizes );

   // Setup test data
   TNL::Containers::Vector< ValueType, DeviceType > inclusive_result( segments.getStorageSize() ),
      exclusive_result( segments.getStorageSize() ), data( segments.getStorageSize() );
   auto data_view = data.getView();
   auto inclusive_result_view = inclusive_result.getView();
   auto exclusive_result_view = exclusive_result.getView();
   auto segmentsSizes_view = segmentsSizes.getConstView();
   data_view = 0.0;
   exclusive_result = 0.0;
   inclusive_result = 0.0;

   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < segmentsSizes_view[ segmentIdx ] ) {
            data_view[ globalIdx ] = segmentIdx + 1.0;
            if( segmentIdx % 2 == 0 ) {
               inclusive_result_view[ globalIdx ] = ( segmentIdx + 1.0 ) * ( localIdx + 1.0 );
               exclusive_result_view[ globalIdx ] = inclusive_result_view[ globalIdx ] - data_view[ globalIdx ];
            }
            else {
               inclusive_result_view[ globalIdx ] = data_view[ globalIdx ];
               exclusive_result_view[ globalIdx ] = data_view[ globalIdx ];
            }
         }
      } );

   // Test inclusive scan
   auto predicate = [] __cuda_callable__( IndexType segmentIdx ) -> bool
   {
      return segmentIdx % 2 == 0;
   };
   auto fetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> ValueType
   {
      if( localIdx < segmentsSizes_view[ segmentIdx ] )
         return data_view[ globalIdx ];
      else
         return 0.0;
   };
   auto write = [ = ] __cuda_callable__( IndexType globalIdx, ValueType value ) mutable
   {
      data_view[ globalIdx ] = value;
   };

   TNL::Algorithms::Segments::inclusiveScanAllSegmentsIf( segments, predicate, fetch, TNL::Plus{}, write );

   HostSegments hostSegments;
   hostSegments = segments;
   TNL::Algorithms::Segments::forAllElements(
      hostSegments,
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < segmentsSizes_view.getElement( segmentIdx ) ) {
            EXPECT_EQ( data_view.getElement( globalIdx ), inclusive_result_view.getElement( globalIdx ) )
               << "segmentIdx = " << segmentIdx << ", localIdx = " << localIdx;
         }
      } );

   // Reset data for exclusive scan test
   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < segmentsSizes_view[ segmentIdx ] ) {
            data_view[ globalIdx ] = segmentIdx + 1.0;
         }
      } );

   TNL::Algorithms::Segments::exclusiveScanAllSegmentsIf( segments, predicate, fetch, TNL::Plus{}, write );

   TNL::Algorithms::Segments::forAllElements(
      hostSegments,
      [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
      {
         if( localIdx < segmentsSizes_view.getElement( segmentIdx ) ) {
            EXPECT_EQ( data_view.getElement( globalIdx ), exclusive_result_view.getElement( globalIdx ) )
               << "segmentIdx = " << segmentIdx << ", localIdx = " << localIdx;
         }
      } );
}
