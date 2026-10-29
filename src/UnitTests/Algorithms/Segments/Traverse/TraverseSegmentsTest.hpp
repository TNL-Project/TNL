#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/Segments/traverse.h>
#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Algorithms/Segments/ReductionLaunchConfigurations.h>
#include <TNL/Algorithms/Segments/TraversingLaunchConfigurations.h>
#include <TNL/Math.h>

#include <iostream>
#include <gtest/gtest.h>

#include "../HostSegmentsGetter.h"

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

   for( const auto& [ launch_config, tag ] : TNL::Algorithms::Segments::traversingLaunchConfigurations< Segments >( segments ) )
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
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using HostVectorType = TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType >;

   const IndexType segmentsCount = 50;
   const IndexType segmentSize = 5;

   VectorType segmentsSizes( segmentsCount );
   segmentsSizes = segmentSize;
   Segments segments( segmentsSizes );
   typename HostSegmentsGetter< Segments >::type hostSegments;
   hostSegments = segments;

   for( const auto& [ launch_config, tag ] : TNL::Algorithms::Segments::traversingLaunchConfigurations< Segments >( segments ) )
   {
      SCOPED_TRACE( tag );

      VectorType v( segments.getStorageSize() );
      HostVectorType host_v;
      auto v_view = v.getView();
      TNL::Algorithms::Segments::forAllElements(
         segments,
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
         {
            v_view[ globalIdx ] = segmentIdx + localIdx;
         },
         launch_config );

      host_v = v;
      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < segmentSize; localIdx++ )
            EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], segmentIdx + localIdx )
               << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
               << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
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

      host_v = v;
      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < segmentSize; localIdx++ )
            if( segmentIdx >= 3 && segmentIdx < segmentsCount - 3 )
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], segmentIdx + localIdx )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
            else
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], 0 );
      }
   }
}

template< typename Segments >
void
test_forElements()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using HostVectorType = TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType >;

   // 288 is a multiple of the slice size of the sliced formats (32), but the number
   // of slices is not a multiple of the slices per block, which tests the last slice
   for( const IndexType segmentsCount : { 280, 288 } ) {
      SCOPED_TRACE( "segmentsCount = " + std::to_string( segmentsCount ) );
      const IndexType maxSegmentSize = 50;

      VectorType segmentsSizes( segmentsCount );
      segmentsSizes.forAllElements(
         [ = ] __cuda_callable__( IndexType idx, IndexType& value )
         {
            value = idx % maxSegmentSize;
         } );

      Segments segments( segmentsSizes );
      typename HostSegmentsGetter< Segments >::type hostSegments;
      hostSegments = segments;
      const TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType > hostSegmentsSizes( segmentsSizes );

      for( const auto& [ launch_config, tag ] :
           TNL::Algorithms::Segments::traversingLaunchConfigurations< Segments >( segments ) )
      {
         SCOPED_TRACE( tag );

         VectorType localIdx_v( segments.getStorageSize(), -1 );
         VectorType globalIdx_v( segments.getStorageSize(), -1 );
         HostVectorType host_localIdx_v;
         HostVectorType host_globalIdx_v;
         auto localIdx_view = localIdx_v.getView();
         auto globalIdx_view = globalIdx_v.getView();
         TNL::Algorithms::Segments::forAllElements(
            segments,
            [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
            {
               localIdx_view[ globalIdx ] = localIdx;
               globalIdx_view[ globalIdx ] = globalIdx;
            },
            launch_config );

         host_localIdx_v = localIdx_v;
         host_globalIdx_v = globalIdx_v;
         for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
            for( IndexType localIdx = 0; localIdx < hostSegmentsSizes[ segmentIdx ]; localIdx++ ) {
               EXPECT_EQ( host_localIdx_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], localIdx )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx )
                  << " segment size = " << hostSegments.getSegmentSize( segmentIdx );
               EXPECT_EQ(
                  host_globalIdx_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ],
                  hostSegments.getGlobalIndex( segmentIdx, localIdx ) )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx )
                  << " segment size = " << hostSegments.getSegmentSize( segmentIdx );
            }
         }

         // Test with segments view and just part of segments
         VectorType v( segments.getStorageSize(), 0 );
         HostVectorType host_v;
         auto v_view = v.getView();
         TNL::Algorithms::Segments::forElements(
            segments.getView(),
            3,
            segmentsCount - 3,
            [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
            {
               v_view[ globalIdx ] = segmentIdx + localIdx;
            },
            launch_config );

         host_v = v;
         for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
            for( IndexType localIdx = 0; localIdx < hostSegmentsSizes[ segmentIdx ]; localIdx++ )
               if( segmentIdx >= 3 && segmentIdx < segmentsCount - 3 )
                  EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], segmentIdx + localIdx )
                     << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                     << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
               else
                  EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], 0 ) << "segmentIdx = " << segmentIdx
                     << " localIdx = " << localIdx << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
         }

         // Test when calling the lambda function without the local index
         TNL::Algorithms::Segments::forAllElements(
            segments,
            [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType globalIdx ) mutable
            {
               v_view[ globalIdx ] = segmentIdx;
            },
            launch_config );

         host_v = v;
         for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
            for( IndexType localIdx = 0; localIdx < hostSegmentsSizes[ segmentIdx ]; localIdx++ )
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], segmentIdx )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
         }
      }
   }
}

// Segments longer than the block size (256 threads), longer than the warp size and
// shorter than the warp size are traversed differently by some launch configurations,
// e.g. the Tiered mapping of CSR. Each element must be visited exactly once, with the
// index of its own segment and its local index.
template< typename Segments >
void
test_forElements_LongSegments()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using HostVectorType = TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType >;

   const IndexType segmentsCount = 300;
   const HostVectorType sizesPattern{ 600, 7, 100, 0, 300, 33, 257, 32 };
   HostVectorType hostSegmentsSizes( segmentsCount );
   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ )
      hostSegmentsSizes[ segmentIdx ] = sizesPattern[ segmentIdx % sizesPattern.getSize() ];
   const VectorType segmentsSizes( hostSegmentsSizes );

   Segments segments( segmentsSizes );
   typename HostSegmentsGetter< Segments >::type hostSegments;
   hostSegments = segments;

   for( const auto& [ launch_config, tag ] : TNL::Algorithms::Segments::traversingLaunchConfigurations< Segments >( segments ) )
   {
      SCOPED_TRACE( tag );

      VectorType visits_v( segments.getStorageSize(), 0 );
      VectorType segmentIdx_v( segments.getStorageSize(), -1 );
      VectorType localIdx_v( segments.getStorageSize(), -1 );
      auto visits_view = visits_v.getView();
      auto segmentIdx_view = segmentIdx_v.getView();
      auto localIdx_view = localIdx_v.getView();
      TNL::Algorithms::Segments::forAllElements(
         segments,
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
         {
            TNL::Algorithms::AtomicOperations< DeviceType >::add( visits_view[ globalIdx ], IndexType( 1 ) );
            segmentIdx_view[ globalIdx ] = segmentIdx;
            localIdx_view[ globalIdx ] = localIdx;
         },
         launch_config );

      const HostVectorType host_visits_v( visits_v );
      const HostVectorType host_segmentIdx_v( segmentIdx_v );
      const HostVectorType host_localIdx_v( localIdx_v );
      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < hostSegmentsSizes[ segmentIdx ]; localIdx++ ) {
            const IndexType globalIdx = hostSegments.getGlobalIndex( segmentIdx, localIdx );
            ASSERT_EQ( host_visits_v[ globalIdx ], 1 ) << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx;
            ASSERT_EQ( host_segmentIdx_v[ globalIdx ], segmentIdx ) << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx;
            ASSERT_EQ( host_localIdx_v[ globalIdx ], localIdx ) << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx;
         }
      }
   }
}

template< typename Segments >
void
test_forElementsIf()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using HostVectorType = TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType >;

   const IndexType segmentsCount = 260;
   const IndexType maxSegmentSize = 50;

   VectorType segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType& value )
      {
         value = idx % maxSegmentSize;
      } );

   Segments segments( segmentsSizes );
   typename HostSegmentsGetter< Segments >::type hostSegments;
   hostSegments = segments;
   const TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType > hostSegmentsSizes( segmentsSizes );

   for( const auto& [ launch_config, tag ] : TNL::Algorithms::Segments::traversingLaunchConfigurations< Segments >( segments ) )
   {
      SCOPED_TRACE( tag );

      VectorType v( segments.getStorageSize(), -1 );
      HostVectorType host_v;
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

      host_v = v;
      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < hostSegmentsSizes[ segmentIdx ]; localIdx++ ) {
            if( segmentIdx % 2 == 0 )
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], segmentIdx + localIdx )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
            else
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], -1 ) << "segmentIdx = " << segmentIdx
                  << " localIdx = " << localIdx << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
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

      host_v = v;
      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < hostSegmentsSizes[ segmentIdx ]; localIdx++ ) {
            if( segmentIdx % 2 == 0 )
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], segmentIdx + localIdx )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );

            else
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], -1 ) << "segmentIdx = " << segmentIdx
                  << " localIdx = " << localIdx << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
         }
      }

      // Test with forSelectedElements
      v = -1;
      TNL::Algorithms::Segments::detail::TraversingOperationsBase< Segments >::forSelectedElements(
         segments.getConstView(),
         0,
         segments.getSegmentCount(),
         [ = ] __cuda_callable__( const IndexType segmentIdx ) -> bool
         {
            return segmentIdx % 2 == 0;
         },
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
         {
            v_view[ globalIdx ] = segmentIdx + localIdx;
         },
         launch_config );

      host_v = v;
      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < hostSegmentsSizes[ segmentIdx ]; localIdx++ ) {
            if( segmentIdx % 2 == 0 )
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], segmentIdx + localIdx )
                  << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );

            else
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], -1 ) << "segmentIdx = " << segmentIdx
                  << " localIdx = " << localIdx << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
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
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   const IndexType segmentsCount = 260;

   VectorType segmentsSizes( segmentsCount, 0 );
   Segments segments( segmentsSizes );

   for( const auto& [ launch_config, tag ] : TNL::Algorithms::Segments::traversingLaunchConfigurations< Segments >( segments ) )
   {
      SCOPED_TRACE( tag );

      VectorType v( segments.getStorageSize(), -1 );
      VectorType segmentIndexes( segmentsCount, 0 );
      segmentIndexes.forElements(
         0,
         segmentsCount / 2,
         [ = ] __cuda_callable__( IndexType idx, IndexType& value )
         {
            value = 2 * idx;
         } );
      auto v_view = v.getView();
      TNL::Algorithms::Segments::forElements(
         segments,
         segmentIndexes.getConstView( 0, segmentsCount / 2 ),
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
         segmentIndexes.getConstView( 0, segmentsCount / 2 ),
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
         segmentIndexes.getConstView( 0, segmentsCount / 2 ),
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
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using HostVectorType = TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType >;

   const IndexType segmentsCount = 260;
   const IndexType maxSegmentSize = 5;

   VectorType segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType& value )
      {
         value = idx % maxSegmentSize;
      } );

   Segments segments( segmentsSizes );
   typename HostSegmentsGetter< Segments >::type hostSegments;
   hostSegments = segments;
   const TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType > hostSegmentsSizes( segmentsSizes );

   for( const auto& [ launch_config, tag ] : TNL::Algorithms::Segments::traversingLaunchConfigurations< Segments >( segments ) )
   {
      SCOPED_TRACE( tag );

      VectorType v( segments.getStorageSize(), -1 );
      VectorType segmentIndexes( segmentsCount, 0 );
      HostVectorType host_v;
      segmentIndexes.forElements(
         0,
         segmentsCount / 2,
         [ = ] __cuda_callable__( IndexType idx, IndexType& value )
         {
            value = 2 * idx;
         } );
      auto v_view = v.getView();
      TNL::Algorithms::Segments::forElements(
         segments,
         segmentIndexes.getConstView( 0, segmentsCount / 2 ),
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
         {
            v_view[ globalIdx ] = segmentIdx + localIdx;
         },
         launch_config );

      host_v = v;
      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < hostSegmentsSizes[ segmentIdx ]; localIdx++ ) {
            if( segmentIdx % 2 == 0 )
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], segmentIdx + localIdx )
                  << "Segment index = " << segmentIdx << " localIdx = " << localIdx
                  << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
            else
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], -1 ) << "Segment index = " << segmentIdx
                  << " localIdx = " << localIdx << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
         }
      }

      // Test with segments view
      v = -1;
      TNL::Algorithms::Segments::forElements(
         segments.getView(),
         segmentIndexes.getConstView( 0, segmentsCount / 2 ),
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
         {
            v_view[ globalIdx ] = segmentIdx + localIdx;
         },
         launch_config );

      host_v = v;
      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < hostSegmentsSizes[ segmentIdx ]; localIdx++ ) {
            if( segmentIdx % 2 == 0 )
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], segmentIdx + localIdx )
                  << "Segment index = " << segmentIdx << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
            else
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], -1 ) << "Segment index = " << segmentIdx
                  << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
         }
      }

      // Test when calling the lambda function without the local index
      v = -1;
      TNL::Algorithms::Segments::forElements(
         segments,
         segmentIndexes.getConstView( 0, segmentsCount / 2 ),
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType globalIdx ) mutable
         {
            v_view[ globalIdx ] = segmentIdx;
         },
         launch_config );

      host_v = v;
      for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
         for( IndexType localIdx = 0; localIdx < hostSegmentsSizes[ segmentIdx ]; localIdx++ ) {
            if( segmentIdx % 2 == 0 )
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], segmentIdx )
                  << "globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
            else
               EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], -1 )
                  << "globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
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
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using HostVectorType = TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType >;

   const IndexType segmentsCount = 16;
   const IndexType maxSegmentSize = 5;

   VectorType segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType& value )
      {
         value = idx % maxSegmentSize;
      } );

   Segments segments( segmentsSizes );
   typename HostSegmentsGetter< Segments >::type hostSegments;
   hostSegments = segments;
   const TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType > hostSegmentsSizes( segmentsSizes );

   VectorType v( segments.getStorageSize() );
   HostVectorType host_v;
   auto v_view = v.getView();
   TNL::Algorithms::Segments::forAllSegments(
      segments,
      [ = ] __cuda_callable__( const SegmentView segment_view ) mutable
      {
         for( IndexType localIdx = 0; localIdx < segment_view.getSize(); localIdx++ ) {
            const IndexType globalIdx = segment_view.getGlobalIndex( localIdx );
            v_view[ globalIdx ] = segment_view.getSegmentIndex() + localIdx;
         }
      } );

   host_v = v;
   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ )
      for( IndexType localIdx = 0; localIdx < hostSegmentsSizes[ segmentIdx ]; localIdx++ ) {
         EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], segmentIdx + localIdx )
            << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
            << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
      }

   // Test with segments view
   v = 0;
   TNL::Algorithms::Segments::forAllSegments(
      segments.getView(),
      [ = ] __cuda_callable__( const SegmentView segment_view ) mutable
      {
         for( IndexType localIdx = 0; localIdx < segment_view.getSize(); localIdx++ ) {
            const IndexType globalIdx = segment_view.getGlobalIndex( localIdx );
            v_view[ globalIdx ] = segment_view.getSegmentIndex() + localIdx;
         }
      } );

   host_v = v;
   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < hostSegmentsSizes[ segmentIdx ]; localIdx++ )
         EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], segmentIdx + localIdx );
   }
}
template< typename Segments >
void
test_forSegmentsWithIndexes()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;
   using SegmentView = typename Segments::SegmentViewType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using HostVectorType = TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType >;

   const IndexType segmentsCount = 16;
   const IndexType maxSegmentSize = 5;

   VectorType segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType& value )
      {
         value = idx % maxSegmentSize;
      } );

   Segments segments( segmentsSizes );
   typename HostSegmentsGetter< Segments >::type hostSegments;
   hostSegments = segments;
   const TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType > hostSegmentsSizes( segmentsSizes );

   VectorType v( segments.getStorageSize() );
   HostVectorType host_v;
   auto v_view = v.getView();

   VectorType segmentIndexes( segmentsCount / 2 );
   segmentIndexes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType& value )
      {
         value = 2 * idx;
      } );

   v = 0;
   TNL::Algorithms::Segments::forSegments(
      segments,
      segmentIndexes,
      [ = ] __cuda_callable__( const SegmentView segment_view ) mutable
      {
         for( IndexType localIdx = 0; localIdx < segment_view.getSize(); localIdx++ ) {
            const IndexType globalIdx = segment_view.getGlobalIndex( localIdx );
            v_view[ globalIdx ] = segment_view.getSegmentIndex() + localIdx;
         }
      } );
   host_v = v;
   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < hostSegmentsSizes[ segmentIdx ]; localIdx++ ) {
         if( segmentIdx % 2 == 0 )
            EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], segmentIdx + localIdx )
               << "segmentIdx = " << segmentIdx << " localIdx = " << localIdx
               << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
         else
            EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], 0 ) << "segmentIdx = " << segmentIdx
               << " localIdx = " << localIdx << " globalIdx = " << hostSegments.getGlobalIndex( segmentIdx, localIdx );
      }
   }
}
template< typename Segments >
void
test_forSegmentsIf()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;
   using SegmentView = typename Segments::SegmentViewType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using HostVectorType = TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType >;

   const IndexType segmentsCount = 260;
   const IndexType maxSegmentSize = 50;

   VectorType segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType& value )
      {
         value = idx % maxSegmentSize;
      } );

   Segments segments( segmentsSizes );

   VectorType v( segments.getStorageSize() );
   HostVectorType host_v;
   auto v_view = v.getView();

   v = 0;
   TNL::Algorithms::Segments::forAllSegmentsIf(
      segments.getView(),
      [ = ] __cuda_callable__( const IndexType segmentIdx ) -> bool
      {
         return segmentIdx % 2 == 0;
      },
      [ = ] __cuda_callable__( const SegmentView segment_view ) mutable
      {
         for( IndexType localIdx = 0; localIdx < segment_view.getSize(); localIdx++ ) {
            const IndexType globalIdx = segment_view.getGlobalIndex( localIdx );
            v_view[ globalIdx ] = segment_view.getSegmentIndex() + localIdx;
         }
      } );
}
template< typename Segments >
void
test_forSegmentsSequential()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;
   using SegmentView = typename Segments::SegmentViewType;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using HostVectorType = TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType >;

   const IndexType segmentsCount = 260;
   const IndexType maxSegmentSize = 50;

   VectorType segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType& value )
      {
         value = idx % maxSegmentSize;
      } );

   Segments segments( segmentsSizes );
   typename HostSegmentsGetter< Segments >::type hostSegments;
   hostSegments = segments;
   const TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType > hostSegmentsSizes( segmentsSizes );

   VectorType v( segments.getStorageSize() );
   HostVectorType host_v;
   auto v_view = v.getView();

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

   host_v = v;
   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      for( IndexType localIdx = 0; localIdx < hostSegmentsSizes[ segmentIdx ]; localIdx++ )
         EXPECT_EQ( host_v[ hostSegments.getGlobalIndex( segmentIdx, localIdx ) ], segmentIdx + localIdx );
   }
}
