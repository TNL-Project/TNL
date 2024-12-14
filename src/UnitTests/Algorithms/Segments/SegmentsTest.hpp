#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/Segments/operations.h>
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
test_forElements_EqualSizes()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 50;
   const IndexType segmentSize = 5;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes = segmentSize;

   Segments segments( segmentsSizes );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize() );
   auto v_view = v.getView();
   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
      {
         v_view[ globalIdx ] = segmentIdx + localIdx;
      } );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentSize; localIdx++ )
         EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx );
   }

   // Test with segments view
   v = 0;
   TNL::Algorithms::Segments::forAllElements(
      segments.getView(),
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
      {
         v_view[ globalIdx ] = segmentIdx + localIdx;
      } );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentSize; localIdx++ )
         EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx );
   }
}

template< typename Segments >
void
test_forElements()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 260;
   const IndexType maxSegmentSize = 70;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx % maxSegmentSize;
      } );

   Segments segments( segmentsSizes );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize() );
   auto v_view = v.getView();
   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
      {
         v_view[ globalIdx ] = segmentIdx + localIdx;
      } );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ )
         EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx );
   }

   // Test with segments view
   v = 0;
   TNL::Algorithms::Segments::forAllElements(
      segments.getView(),
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
      {
         v_view[ globalIdx ] = segmentIdx + localIdx;
      } );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ )
         EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx );
   }

   // Test when calling the lambda function without the local index
   TNL::Algorithms::Segments::forAllElements(
      segments,
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType globalIdx ) mutable
      {
         v_view[ globalIdx ] = segmentIdx;
      } );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ )
         EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx )
            << "globalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
   }
}

template< typename Segments >
void
test_forElementsIf()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 260;
   const IndexType maxSegmentSize = 70;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx % maxSegmentSize;
      } );

   Segments segments( segmentsSizes );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize(), -1 );
   auto v_view = v.getView();
   TNL::Algorithms::Segments::forAllElementsIf(
      segments,
      [ = ] __cuda_callable__( const IndexType segmentIdx ) -> bool
      {
         return segmentIdx % 2 == 0;
      },
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
      {
         v_view[ globalIdx ] = segmentIdx + localIdx;
      } );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ ) {
         if( segmentIdx % 2 == 0 )
            EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx );
         else
            EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), -1 );
      }
   }

   // Test with segments view
   v = -1;
   TNL::Algorithms::Segments::forAllElementsIf(
      segments.getView(),
      [ = ] __cuda_callable__( const IndexType segmentIdx ) -> bool
      {
         return segmentIdx % 2 == 0;
      },
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
      {
         v_view[ globalIdx ] = segmentIdx + localIdx;
      } );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ ) {
         if( segmentIdx % 2 == 0 )
            EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx );
         else
            EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), -1 );
      }
   }
}

template< typename Segments >
void
test_forElementsWithSegmentIndexes()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 260;
   const IndexType maxSegmentSize = 70;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx % maxSegmentSize;
      } );

   Segments segments( segmentsSizes );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize(), -1 ),
      segmentIndexes( segmentsCount, 0 );
   segmentIndexes.forElements( 0,
                               segmentsCount / 2,
                               [ = ] __cuda_callable__( IndexType idx, IndexType & value )
                               {
                                  value = 2 * idx;
                               } );
   auto v_view = v.getView();
   TNL::Algorithms::Segments::forElements(
      segments,
      segmentIndexes,
      0,
      segmentsCount / 2,
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
      {
         v_view[ globalIdx ] = segmentIdx + localIdx;
      } );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ ) {
         if( segmentIdx % 2 == 0 )
            EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx )
               << "Segment index = " << segmentIdx << " gblobalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
         else
            EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), -1 )
               << "Segment index = " << segmentIdx << " gblobalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
      }
   }

   // Test with segments view
   v = -1;
   TNL::Algorithms::Segments::forElements(
      segments.getView(),
      segmentIndexes,
      0,
      segmentsCount / 2,
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
      {
         v_view[ globalIdx ] = segmentIdx + localIdx;
      } );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ ) {
         if( segmentIdx % 2 == 0 )
            EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx )
               << "Segment index = " << segmentIdx << " gblobalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
         else
            EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), -1 )
               << "Segment index = " << segmentIdx << " gblobalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
      }
   }

   // Test when calling the lambda function without the local index
   v = -1;
   TNL::Algorithms::Segments::forElements(
      segments,
      segmentIndexes,
      0,
      segmentsCount / 2,
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType globalIdx ) mutable
      {
         v_view[ globalIdx ] = segmentIdx;
      } );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ ) {
         if( segmentIdx % 2 == 0 )
            EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx )
               << "globalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
         else
            EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), -1 )
               << "globalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
      }
   }
}

template< typename Segments >
void
test_forSegments()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;
   using SegmentView = typename Segments::SegmentViewType;

   const IndexType segmentsCount = 260;
   const IndexType maxSegmentSize = 70;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx % maxSegmentSize;
      } );

   Segments segments( segmentsSizes );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize() );
   auto v_view = v.getView();
   TNL::Algorithms::Segments::forAllSegments( segments,
                                              [ = ] __cuda_callable__( const SegmentView segment_view ) mutable
                                              {
                                                 for( IndexType localIdx = 0; localIdx < segment_view.getSize(); localIdx++ ) {
                                                    const IndexType globalIdx = segment_view.getGlobalIndex( localIdx );
                                                    v_view[ globalIdx ] = segment_view.getSegmentIndex() + localIdx;
                                                 }
                                              } );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ )
         EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx );
   }

   // Test with segments view
   v = 0;
   TNL::Algorithms::Segments::forAllSegments( segments.getView(),
                                              [ = ] __cuda_callable__( const SegmentView segment_view ) mutable
                                              {
                                                 for( IndexType localIdx = 0; localIdx < segment_view.getSize(); localIdx++ ) {
                                                    const IndexType globalIdx = segment_view.getGlobalIndex( localIdx );
                                                    v_view[ globalIdx ] = segment_view.getSegmentIndex() + localIdx;
                                                 }
                                              } );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ )
         EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx );
   }

   v = 0;
   TNL::Algorithms::Segments::sequentialForAllSegments(
      segments,
      [ = ] __cuda_callable__( const SegmentView segment_view ) mutable
      {
         for( IndexType localIdx = 0; localIdx < segment_view.getSize(); localIdx++ ) {
            const IndexType globalIdx = segment_view.getGlobalIndex( localIdx );
            v_view[ globalIdx ] = segment_view.getSegmentIndex() + localIdx;
         }
      } );

   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ )
         EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx );
   }
}

template< typename Segments, typename SegmentsReductionKernel >
void
test_reduceAllSegments_MaximumInSegments()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 20;
   const IndexType segmentSize = 5;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes = segmentSize;

   Segments segments( segmentsSizes );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize() );

   auto view = v.getView();
   auto init =
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable -> bool
   {
      TNL_ASSERT_LT( globalIdx, view.getSize(), "" );
      view[ globalIdx ] = segmentIdx * 5 + localIdx + 1;
      return true;
   };
   TNL::Algorithms::Segments::forAllElements( segments, init );

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > result( segmentsCount );

   const auto v_view = v.getConstView();
   auto result_view = result.getView();
   auto fetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> IndexType
   {
      // some segments may use padding zeros and their size may be greater than the original segment size
      if( localIdx < segmentSize )
         return v_view[ globalIdx ];
      return 0;
   };
   auto reduce = [] __cuda_callable__( IndexType & a, const IndexType b ) -> IndexType
   {
      return TNL::max( a, b );
   };
   auto keep = [ = ] __cuda_callable__( const IndexType i, const IndexType a ) mutable
   {
      result_view[ i ] = a;
   };
   SegmentsReductionKernel reductionKernel;
   reductionKernel.init( segments );
   reductionKernel.reduceAllSegments( segments, fetch, reduce, keep, std::numeric_limits< IndexType >::min() );

   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( result.getElement( i ), ( i + 1 ) * segmentSize );

   result_view = 0;
   reductionKernel.reduceAllSegments( segments.getView(), fetch, reduce, keep, std::numeric_limits< IndexType >::min() );
   for( IndexType i = 0; i < segmentsCount; i++ )
      EXPECT_EQ( result.getElement( i ), ( i + 1 ) * segmentSize );
}
