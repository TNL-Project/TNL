#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/Segments/traverse.h>
#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Algorithms/Segments/ReductionLaunchConfigurations.h>
#include <TNL/Algorithms/Segments/TraversingLaunchConfigurations.h>
#include <TNL/Math.h>

#include <iostream>
#include <gtest/gtest.h>

template< typename Segments >
void
test_reduceSegments_MaximumInSegments()
{
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
test_reduceSegments_MaximumInSegments_short_fetch()
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
test_reduceSegmentsWithArgument_MaximumInSegments()
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
      auto keep = [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType res ) mutable
      {
         result_view[ segmentIdx ] = res;
         args_view[ segmentIdx ] = localIdx;
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

template< typename Segments >
void
test_reduceSegmentsWithSegmentIndexes_MaximumInSegments()
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

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentIndexes( ( segmentsCount + 1 ) / 2 );
   segmentIndexes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = 2 * idx;
      } );

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

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > result( segmentsCount, -1 );

      const auto v_view = v.getConstView();
      auto result_view = result.getView();
      auto fetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> IndexType
      {
         if( v_view[ globalIdx ] >= 0 )
            return v_view[ globalIdx ];
         return 0;
      };
      auto keep =
         [ = ] __cuda_callable__( const IndexType indexOfSegmentIdx, const IndexType segmentIdx, const IndexType res ) mutable
      {
         result_view[ segmentIdx ] = res;
      };
      TNL::Algorithms::Segments::reduceSegments( segments, segmentIndexes, fetch, TNL::Max{}, keep, launch_config );

      for( IndexType i = 0; i < segmentsCount; i++ ) {
         if( i % 2 == 0 )
            EXPECT_EQ( result.getElement( i ), i % maxSegmentSize + 1 ) << "segmentIdx = " << i;
         else
            EXPECT_EQ( result.getElement( i ), -1 ) << "segmentIdx = " << i;
      }

      // Test with segments view and short fetch
      result_view = -1;
      auto short_fetch = [ = ] __cuda_callable__( IndexType globalIdx ) -> IndexType
      {
         if( v_view[ globalIdx ] >= 0 )
            return v_view[ globalIdx ];
         return 0;
      };

      TNL::Algorithms::Segments::reduceSegments(
         segments.getView(), segmentIndexes, short_fetch, TNL::Max{}, keep, launch_config );

      for( IndexType i = 0; i < segmentsCount; i++ ) {
         if( i % 2 == 0 )
            EXPECT_EQ( result.getElement( i ), i % maxSegmentSize + 1 ) << "segmentIdx = " << i;
         else
            EXPECT_EQ( result.getElement( i ), -1 ) << "segmentIdx = " << i;
      }
   }
}

template< typename Segments >
void
test_reduceSegmentsWithSegmentIndexesAndArgument_MaximumInSegments()
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

   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentIndexes( ( segmentsCount + 1 ) / 2 );
   segmentIndexes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = 2 * idx;
      } );

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

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > result( segmentsCount, -1 ), args( segmentsCount, -1 );

      const auto v_view = v.getConstView();
      auto result_view = result.getView();
      auto args_view = args.getView();
      auto fetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> IndexType
      {
         if( v_view[ globalIdx ] >= 0 )
            return v_view[ globalIdx ];
         return 0;
      };
      auto keep = [ = ] __cuda_callable__( const IndexType indexOfSegmentIdx,
                                           const IndexType segmentIdx,
                                           const IndexType localIdx,
                                           const IndexType res ) mutable
      {
         result_view[ segmentIdx ] = res;
         args_view[ segmentIdx ] = localIdx;
      };
      TNL::Algorithms::Segments::reduceSegmentsWithArgument(
         segments, segmentIndexes, fetch, TNL::MaxWithArg{}, keep, launch_config );

      for( IndexType i = 0; i < segmentsCount; i++ ) {
         if( i % 2 == 0 ) {
            EXPECT_EQ( result.getElement( i ), i % maxSegmentSize + 1 ) << "segmentIdx = " << i;
            EXPECT_EQ( args.getElement( i ), i % maxSegmentSize ) << "segmentIdx = " << i;
         }
         else {
            EXPECT_EQ( result.getElement( i ), -1 ) << "segmentIdx = " << i;
            EXPECT_EQ( args.getElement( i ), -1 ) << "segmentIdx = " << i;
         }
      }

      // Test with segments view and short fetch
      result_view = -1;
      args_view = -1;
      auto short_fetch = [ = ] __cuda_callable__( IndexType globalIdx ) -> IndexType
      {
         if( v_view[ globalIdx ] >= 0 )
            return v_view[ globalIdx ];
         return 0;
      };

      TNL::Algorithms::Segments::reduceSegmentsWithArgument(
         segments.getView(), segmentIndexes, short_fetch, TNL::MaxWithArg{}, keep, launch_config );

      for( IndexType i = 0; i < segmentsCount; i++ ) {
         if( i % 2 == 0 ) {
            EXPECT_EQ( result.getElement( i ), i % maxSegmentSize + 1 ) << "segmentIdx = " << i;
            EXPECT_EQ( args.getElement( i ), i % maxSegmentSize ) << "segmentIdx = " << i;
         }
         else {
            EXPECT_EQ( result.getElement( i ), -1 ) << "segmentIdx = " << i;
            EXPECT_EQ( args.getElement( i ), -1 ) << "segmentIdx = " << i;
         }
      }
   }
}

template< typename Segments >
void
test_reduceSegmentsIf_MaximumInSegments()
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

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > result( segmentsCount, -1 );

      const auto v_view = v.getConstView();
      auto result_view = result.getView();
      auto condition = [ = ] __cuda_callable__( IndexType segmentIdx ) -> bool
      {
         return segmentIdx % 2 == 0;
      };
      auto fetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> IndexType
      {
         if( v_view[ globalIdx ] >= 0 )
            return v_view[ globalIdx ];
         return 0;
      };
      auto keep =
         [ = ] __cuda_callable__( const IndexType indexOfSegmentIdx, const IndexType segmentIdx, const IndexType res ) mutable
      {
         result_view[ segmentIdx ] = res;
      };
      TNL::Algorithms::Segments::reduceSegmentsIf(
         segments, 0, segmentsCount, condition, fetch, TNL::Max{}, keep, launch_config );

      for( IndexType i = 0; i < segmentsCount; i++ ) {
         if( i % 2 == 0 )
            EXPECT_EQ( result.getElement( i ), i % maxSegmentSize + 1 ) << "segmentIdx = " << i;
         else
            EXPECT_EQ( result.getElement( i ), -1 ) << "segmentIdx = " << i;
      }

      // Test with segments view and short fetch
      result_view = -1;
      auto short_fetch = [ = ] __cuda_callable__( IndexType globalIdx ) -> IndexType
      {
         if( v_view[ globalIdx ] >= 0 )
            return v_view[ globalIdx ];
         return 0;
      };

      TNL::Algorithms::Segments::reduceAllSegmentsIf(
         segments.getView(), condition, short_fetch, TNL::Max{}, keep, launch_config );

      for( IndexType i = 0; i < segmentsCount; i++ ) {
         if( i % 2 == 0 )
            EXPECT_EQ( result.getElement( i ), i % maxSegmentSize + 1 ) << "segmentIdx = " << i;
         else
            EXPECT_EQ( result.getElement( i ), -1 ) << "segmentIdx = " << i;
      }
   }
}

template< typename Segments >
void
test_reduceSegmentsIfWithArgument_MaximumInSegments()
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

      TNL::Containers::Vector< IndexType, DeviceType, IndexType > result( segmentsCount, -1 ), args( segmentsCount, -1 );

      const auto v_view = v.getConstView();
      auto result_view = result.getView();
      auto args_view = args.getView();
      auto condition = [ = ] __cuda_callable__( IndexType segmentIdx ) -> bool
      {
         return segmentIdx % 2 == 0;
      };
      auto fetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> IndexType
      {
         if( v_view[ globalIdx ] >= 0 )
            return v_view[ globalIdx ];
         return 0;
      };
      auto keep = [ = ] __cuda_callable__( const IndexType indexOfSegmentIdx,
                                           const IndexType segmentIdx,
                                           const IndexType localIdx,
                                           const IndexType res ) mutable
      {
         result_view[ segmentIdx ] = res;
         args_view[ segmentIdx ] = localIdx;
      };
      TNL::Algorithms::Segments::reduceAllSegmentsIfWithArgument(
         segments, condition, fetch, TNL::MaxWithArg{}, keep, launch_config );

      for( IndexType i = 0; i < segmentsCount; i++ ) {
         if( i % 2 == 0 ) {
            EXPECT_EQ( result.getElement( i ), i % maxSegmentSize + 1 ) << "segmentIdx = " << i;
            EXPECT_EQ( args.getElement( i ), i % maxSegmentSize ) << "segmentIdx = " << i;
         }
         else {
            EXPECT_EQ( result.getElement( i ), -1 ) << "segmentIdx = " << i;
            EXPECT_EQ( args.getElement( i ), -1 ) << "segmentIdx = " << i;
         }
      }

      // Test with segments view and short fetch
      result_view = -1;
      args_view = -1;
      auto short_fetch = [ = ] __cuda_callable__( IndexType globalIdx ) -> IndexType
      {
         if( v_view[ globalIdx ] >= 0 )
            return v_view[ globalIdx ];
         return 0;
      };

      TNL::Algorithms::Segments::reduceSegmentsIfWithArgument(
         segments.getView(), 0, segmentsCount, condition, short_fetch, TNL::MaxWithArg{}, keep, launch_config );

      for( IndexType i = 0; i < segmentsCount; i++ ) {
         if( i % 2 == 0 ) {
            EXPECT_EQ( result.getElement( i ), i % maxSegmentSize + 1 ) << "segmentIdx = " << i;
            EXPECT_EQ( args.getElement( i ), i % maxSegmentSize ) << "segmentIdx = " << i;
         }
         else {
            EXPECT_EQ( result.getElement( i ), -1 ) << "segmentIdx = " << i;
            EXPECT_EQ( args.getElement( i ), -1 ) << "segmentIdx = " << i;
         }
      }
   }
}
