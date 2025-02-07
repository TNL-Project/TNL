#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/Segments/traverse.h>
#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Algorithms/Segments/ReductionLaunchConfigurations.h>
#include <TNL/Algorithms/Segments/TraversingLaunchConfigurations.h>
#include <TNL/Math.h>

#include <iostream>
#include <gtest/gtest.h>

#define UNDEF
#ifdef UNDEF

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
test_forElements_EmptySegments()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 50;
   const IndexType segmentSize = 0;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes = segmentSize;

   Segments segments( segmentsSizes );

   for( auto [ launch_config, tag ] :
        TNL::Algorithms::Segments::TraversingLaunchConfigurations< Segments >::create( segments ) )
   {
      SCOPED_TRACE( tag );

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize(), -1 );
      auto v_view = v.getView();
      TNL::Algorithms::Segments::forAllElements(
         segments,
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
         {
            v_view[ globalIdx ] = segmentIdx + localIdx;
         },
         launch_config );

      EXPECT_TRUE( TNL::all( TNL::equalTo( v, -1 ) ) );

      // Test with segments view and just part of segments
      v = -1;
      TNL::Algorithms::Segments::forElements(
         segments.getView(),
         3,
         segmentsCount - 3,
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
         {
            v_view[ globalIdx ] = segmentIdx + localIdx;
         },
         launch_config );
      EXPECT_TRUE( TNL::all( TNL::equalTo( v, -1 ) ) );
   }
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

   for( auto [ launch_config, tag ] :
        TNL::Algorithms::Segments::TraversingLaunchConfigurations< Segments >::create( segments ) )
   {
      SCOPED_TRACE( tag );

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize() );
      auto v_view = v.getView();
      TNL::Algorithms::Segments::forAllElements(
         segments,
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
         {
            v_view[ globalIdx ] = segmentIdx + localIdx;
         },
         launch_config );

      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < segmentSize; localIdx++ )
            EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx )
               << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
               << " globalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
      }

      // Test with segments view and just part of segments
      v = 0;
      TNL::Algorithms::Segments::forElements(
         segments.getView(),
         3,
         segmentsCount - 3,
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
         {
            v_view[ globalIdx ] = segmentIdx + localIdx;
         },
         launch_config );

      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < segmentSize; localIdx++ )
            if( segmentIdx >= 3 && segmentIdx < segmentsCount - 3 )
               EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
            else
               EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), 0 );
      }
   }
}

template< typename Segments >
void
test_forElements()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 280;
   const IndexType maxSegmentSize = 50;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx % maxSegmentSize;
      } );

   Segments segments( segmentsSizes );

   for( auto [ launch_config, tag ] :
        TNL::Algorithms::Segments::TraversingLaunchConfigurations< Segments >::create( segments ) )
   {
      SCOPED_TRACE( tag );

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize() );
      auto v_view = v.getView();
      TNL::Algorithms::Segments::forAllElements(
         segments,
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
         {
            v_view[ globalIdx ] = segmentIdx + localIdx;
         },
         launch_config );

      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ )
            EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx )
               << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
               << " globalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
      }

      // Test with segments view and just part of segments
      v = 0;
      TNL::Algorithms::Segments::forElements(
         segments.getView(),
         3,
         segmentsCount - 3,
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
         {
            v_view[ globalIdx ] = segmentIdx + localIdx;
         },
         launch_config );

      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ )
            if( segmentIdx >= 3 && segmentIdx < segmentsCount - 3 )
               EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
            else
               EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), 0 );
      }

      // Test when calling the lambda function without the local index
      TNL::Algorithms::Segments::forAllElements(
         segments,
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType globalIdx ) mutable
         {
            v_view[ globalIdx ] = segmentIdx;
         },
         launch_config );

      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ )
            EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx )
               << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
               << " globalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
      }
   }
}

template< typename Segments >
void
test_forElementsIf()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 260;
   const IndexType maxSegmentSize = 50;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx % maxSegmentSize;
      } );

   Segments segments( segmentsSizes );

   for( auto [ launch_config, tag ] :
        TNL::Algorithms::Segments::TraversingLaunchConfigurations< Segments >::create( segments ) )
   {
      SCOPED_TRACE( tag );

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
         },
         launch_config );

      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ ) {
            if( segmentIdx % 2 == 0 )
               EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
            else
               EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), -1 )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
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
         },
         launch_config );

      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ ) {
            if( segmentIdx % 2 == 0 )
               EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );

            else
               EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), -1 )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
         }
      }

      // Test with forElementsIfSparse
      v = -1;
      TNL::Algorithms::Segments::forAllElementsIfSparse(
         segments.getView(),
         [ = ] __cuda_callable__( const IndexType segmentIdx ) -> bool
         {
            return segmentIdx % 2 == 0;
         },
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
         {
            v_view[ globalIdx ] = segmentIdx + localIdx;
         },
         launch_config );

      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ ) {
            if( segmentIdx % 2 == 0 )
               EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), segmentIdx + localIdx )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );

            else
               EXPECT_EQ( v.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ), -1 )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << segments.getGlobalIndex( segmentIdx, localIdx );
         }
      }
   }
}

template< typename Segments >
void
test_forElementsWithSegmentIndexes_EmptySegments()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 260;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount, 0 );
   Segments segments( segmentsSizes );

   for( auto [ launch_config, tag ] :
        TNL::Algorithms::Segments::TraversingLaunchConfigurations< Segments >::create( segments ) )
   {
      SCOPED_TRACE( tag );

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
         },
         launch_config );
      EXPECT_TRUE( TNL::all( TNL::equalTo( v, -1 ) ) );

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
         },
         launch_config );
      EXPECT_TRUE( TNL::all( TNL::equalTo( v, -1 ) ) );

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
         },
         launch_config );

      EXPECT_TRUE( TNL::all( TNL::equalTo( v, -1 ) ) );
   }
}

template< typename Segments >
void
test_forElementsWithSegmentIndexes()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 260;
   const IndexType maxSegmentSize = 50;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx % maxSegmentSize;
      } );

   Segments segments( segmentsSizes );

   for( auto [ launch_config, tag ] :
        TNL::Algorithms::Segments::TraversingLaunchConfigurations< Segments >::create( segments ) )
   {
      SCOPED_TRACE( tag );

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
         },
         launch_config );

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
         },
         launch_config );

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
         },
         launch_config );

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
}

template< typename Segments >
void
test_forSegments()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;
   using SegmentView = typename Segments::SegmentViewType;

   const IndexType segmentsCount = 260;
   const IndexType maxSegmentSize = 50;

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

#endif

template< typename Segments >
void
test_reduceAllSegments_MaximumInSegments()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 256;
   const IndexType segmentSize = 70;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes = segmentSize;

   Segments segments( segmentsSizes );

   for( auto [ launch_config, tag ] : reductionLaunchConfigurations( segments ) ) {
      SCOPED_TRACE( tag );

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize() );

      auto view = v.getView();
      auto init = [ = ] __cuda_callable__(
                     const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable -> bool
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
      TNL::Algorithms::Segments::reduceAllSegments(
         segments, fetch, reduce, keep, std::numeric_limits< IndexType >::min(), launch_config );

      for( IndexType i = 0; i < segmentsCount; i++ )
         EXPECT_EQ( result.getElement( i ), 5 * i + segmentSize );

      result_view = 0;
      TNL::Algorithms::Segments::reduceAllSegments(
         segments.getView(), fetch, reduce, keep, std::numeric_limits< IndexType >::min(), launch_config );
      for( IndexType i = 0; i < segmentsCount; i++ )
         EXPECT_EQ( result.getElement( i ), 5 * i + segmentSize );
   }
}

template< typename Segments >
void
test_reduceAllSegments_MaximumInSegments_short_fetch()
{
   // This test calls the fetch function only with the globalIdx parameter.
   // It can be used only for segments without padding zeros.
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 270;
   const IndexType segmentSize = 70;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes = segmentSize;

   Segments segments( segmentsSizes );

   for( auto [ launch_config, tag ] : reductionLaunchConfigurations( segments ) ) {
      SCOPED_TRACE( tag );

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize() );
      v = -1;

      auto view = v.getView();
      auto init = [ = ] __cuda_callable__(
                     const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable -> bool
      {
         if( localIdx < segmentSize )
            view[ globalIdx ] = segmentIdx * 5 + localIdx + 1;
         return true;
      };
      TNL::Algorithms::Segments::forAllElements( segments, init );

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > result( segmentsCount );

      const auto v_view = v.getConstView();
      auto result_view = result.getView();
      auto fetch = [ = ] __cuda_callable__( IndexType globalIdx ) -> IndexType
      {
         if( v_view[ globalIdx ] >= 0 )
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
      TNL::Algorithms::Segments::reduceAllSegments(
         segments, fetch, reduce, keep, std::numeric_limits< IndexType >::min(), launch_config );

      for( IndexType i = 0; i < segmentsCount; i++ )
         EXPECT_EQ( result.getElement( i ), 5 * i + segmentSize ) << "segmentIdx = " << i;

      result_view = 0;
      TNL::Algorithms::Segments::reduceAllSegments(
         segments.getView(), fetch, reduce, keep, std::numeric_limits< IndexType >::min(), launch_config );
      for( IndexType i = 0; i < segmentsCount; i++ )
         EXPECT_EQ( result.getElement( i ), 5 * i + segmentSize ) << "segmentIdx = " << i;
   }
}

template< typename Segments >
void
test_reduceAllSegments_MaximumInSegmentsWithArgument()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   const IndexType segmentsCount = 270;
   const IndexType maxSegmentSize = 70;

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx % maxSegmentSize + 1;
      } );

   Segments segments( segmentsSizes );

   for( auto [ launch_config, tag ] : reductionLaunchConfigurations( segments ) ) {
      SCOPED_TRACE( tag );

      if( std::is_same_v< DeviceType, TNL::Devices::Cuda > && std::is_same_v< IndexType, long >
          && TNL::Algorithms::Segments::isCSRSegments_v< Segments >
          && launch_config.getThreadsToSegmentsMapping() == TNL::Algorithms::Segments::ThreadsToSegmentsMapping::UserDefined
          && launch_config.getThreadsPerSegmentCount() > 32 )
         continue;  // TODO: Multivector in CSR does not work for long int on CUDA. Really don't know why. Needs to be fixed.

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > v( segments.getStorageSize() );
      v = -1;

      auto view = v.getView();
      auto init = [ = ] __cuda_callable__(
                     const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable -> bool
      {
         TNL_ASSERT_LT( globalIdx, view.getSize(), "" );
         // some segments may use padding zeros and their size may be greater than the original segment size
         if( localIdx <= segmentIdx % maxSegmentSize )
            view[ globalIdx ] = localIdx + 1;
         return true;
      };
      TNL::Algorithms::Segments::forAllElements( segments, init );

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > result( segmentsCount ), args( segmentsCount );

      const auto v_view = v.getConstView();
      auto result_view = result.getView();
      auto args_view = args.getView();
      auto fetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> IndexType
      {
         if( v_view[ globalIdx ] >= 0 )
            return v_view[ globalIdx ];
         return 0;
      };
      auto keep = [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType res, const IndexType arg ) mutable
      {
         result_view[ segmentIdx ] = res;
         args_view[ segmentIdx ] = arg;
      };
      TNL::Algorithms::Segments::reduceAllSegmentsWithArgument( segments, fetch, TNL::MaxWithArg{}, keep, launch_config );

      for( IndexType i = 0; i < segmentsCount; i++ ) {
         EXPECT_EQ( result.getElement( i ), i % maxSegmentSize + 1 ) << "segmentIdx = " << i;
         EXPECT_EQ( args.getElement( i ), i % maxSegmentSize ) << "segmentIdx = " << i;
      }

      // Test with segments view and short fetch
      result_view = 0;
      args_view = 0;
      auto short_fetch = [ = ] __cuda_callable__( IndexType globalIdx ) -> IndexType
      {
         if( v_view[ globalIdx ] >= 0 )
            return v_view[ globalIdx ];
         return 0;
      };

      TNL::Algorithms::Segments::reduceAllSegmentsWithArgument(
         segments.getView(), short_fetch, TNL::MaxWithArg{}, keep, launch_config );

      for( IndexType i = 0; i < segmentsCount; i++ ) {
         EXPECT_EQ( result.getElement( i ), i % maxSegmentSize + 1 ) << "segmentIdx = " << i;
         EXPECT_EQ( args.getElement( i ), i % maxSegmentSize ) << "segmentIdx = " << i;
      }
   }
}
