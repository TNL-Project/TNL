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
          && launch_config.getThreadsToSegmentsMapping() == TNL::Algorithms::Segments::ThreadsToSegmentsMapping::Fixed
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
          && launch_config.getThreadsToSegmentsMapping() == TNL::Algorithms::Segments::ThreadsToSegmentsMapping::Fixed
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
          && launch_config.getThreadsToSegmentsMapping() == TNL::Algorithms::Segments::ThreadsToSegmentsMapping::Fixed
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
          && launch_config.getThreadsToSegmentsMapping() == TNL::Algorithms::Segments::ThreadsToSegmentsMapping::Fixed
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
          && launch_config.getThreadsToSegmentsMapping() == TNL::Algorithms::Segments::ThreadsToSegmentsMapping::Fixed
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

template< typename Segments >
void
test_reduce_SumOfMaximums()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;
   using ValueType = double;

   const IndexType segmentsCount = 270;
   const IndexType maxSegmentSize = 50;

   // Initialize segments with equal sizes
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx % maxSegmentSize + 1;
      } );

   Segments segments( segmentsSizes );

   // Initialize data
   TNL::Containers::Vector< ValueType, DeviceType, IndexType > v( segments.getStorageSize(), -1 );
   auto view = v.getView();
   auto segmentsSizesView = segmentsSizes.getView();
   auto init =
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable -> bool
   {
      TNL_ASSERT_LT( globalIdx, view.getSize(), "" );
      if( localIdx < segmentsSizesView[ segmentIdx ] )
         view[ globalIdx ] = segmentIdx + localIdx + 1;
      return true;
   };
   TNL::Algorithms::Segments::forAllElements( segments, init );

   // Test complete reduction: find sum of maximum values in each segment
   for( auto [ launch_config, tag ] : reductionLaunchConfigurations( segments ) ) {
      SCOPED_TRACE( tag );

      // Define segment fetch and reduction (find maximum in each segment)
      auto segmentFetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> ValueType
      {
         return view[ globalIdx ] != -1 ? view[ globalIdx ]
                                        : std::numeric_limits< ValueType >::lowest();  // Ignore padding zeros
      };

      // Define result fetch and reduction (sum the maximums)
      auto finalFetch = [ = ] __cuda_callable__( const ValueType& value ) -> ValueType
      {
         return value;
      };

      // Perform complete reduction
      const ValueType result = reduceAll( segments, segmentFetch, TNL::Max{}, finalFetch, TNL::Plus{}, launch_config );

      TNL::Containers::Vector< ValueType, DeviceType, IndexType > resultVector( segmentsCount );
      resultVector.forAllElements(
         [ = ] __cuda_callable__( IndexType segmentIdx, ValueType & value )
         {
            value = segmentIdx + segmentIdx % maxSegmentSize + 1;  // Each segment's maximum is (segmentIdx + segmentSize)
         } );
      auto expectedResult = TNL::sum( resultVector );

      EXPECT_NEAR( result, expectedResult, 1e-10 );

      const ValueType result2 = reduce( segments, 10, 100, segmentFetch, TNL::Max{}, finalFetch, TNL::Plus{}, launch_config );
      auto expectedResult2 = TNL::sum( resultVector.getView( 10, 100 ) );
      EXPECT_NEAR( result2, expectedResult2, 1e-10 );
   }
}

template< typename Segments >
void
test_reduce_ProductOfSums()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;
   using ValueType = double;

   const IndexType segmentsCount = 10;  // Using smaller numbers to avoid overflow
   const IndexType maxSegmentSize = 5;

   // Initialize segments with equal sizes
   TNL::Containers::Vector< IndexType, DeviceType, IndexType > segmentsSizes( segmentsCount );
   segmentsSizes.forAllElements(
      [ = ] __cuda_callable__( IndexType idx, IndexType & value )
      {
         value = idx % maxSegmentSize + 1;
      } );
   Segments segments( segmentsSizes );

   // Initialize data
   TNL::Containers::Vector< ValueType, DeviceType, IndexType > v( segments.getStorageSize(), -1 );
   auto view = v.getView();
   auto segmentsSizesView = segmentsSizes.getView();
   auto init =
      [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable -> bool
   {
      TNL_ASSERT_LT( globalIdx, view.getSize(), "" );
      if( localIdx < segmentsSizesView[ segmentIdx ] )
         view[ globalIdx ] = segmentIdx + 1.0;  // All elements in a segment have same value
      return true;
   };
   TNL::Algorithms::Segments::forAllElements( segments, init );

   // Test complete reduction: find product of sums in each segment
   for( auto [ launch_config, tag ] : reductionLaunchConfigurations( segments ) ) {
      SCOPED_TRACE( tag );

      // Define segment fetch and reduction (sum elements in each segment)
      auto segmentFetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> ValueType
      {
         return view[ globalIdx ] != -1 ? view[ globalIdx ] : 0;
      };

      // Define result fetch and reduction (multiply the sums)
      auto finalFetch = [ = ] __cuda_callable__( const ValueType& value ) -> ValueType
      {
         return value;
      };

      // Perform complete reduction
      const ValueType result = reduceAll( segments, segmentFetch, TNL::Plus{}, finalFetch, TNL::Multiplies{}, launch_config );

      TNL::Containers::Vector< ValueType, DeviceType, IndexType > resultVector( segmentsCount );
      resultVector.forAllElements(
         [ = ] __cuda_callable__( IndexType segmentIdx, ValueType & value )
         {
            value = ( segmentIdx + 1 )
                  * ( segmentIdx % maxSegmentSize + 1 );  // Each segment's sum is (segmentIdx + 1) * segmentSize
         } );

      // Each segment's sum is (segmentIdx + 1) * segmentSize
      // The product of these sums should be: product((i + 1) * segmentSize) for i in [0, segmentsCount)
      ValueType expectedResult = TNL::product( resultVector );
      EXPECT_NEAR( result, expectedResult, 1e-10 );

      const ValueType result2 =
         reduce( segments, 2, 8, segmentFetch, TNL::Plus{}, finalFetch, TNL::Multiplies{}, launch_config );
      ValueType expectedResult2 = TNL::product( resultVector.getView( 2, 8 ) );
      EXPECT_NEAR( result2, expectedResult2, 1e-10 );
   }
}
