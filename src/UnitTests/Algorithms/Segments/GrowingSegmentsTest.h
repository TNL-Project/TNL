#include <TNL/Algorithms/Segments/GrowingSegments.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Functional.h>

#include "SegmentsTest.hpp"
#include <iostream>

#include <gtest/gtest.h>

// test fixture for typed tests
template< typename GrowingSegments >
class GrowingSegmentsTest : public ::testing::Test
{
protected:
   using SegmentsType = GrowingSegments;
};

// types for which MatrixTest is instantiated
using GrowingSegmentsTypes = ::testing::Types< TNL::Algorithms::Segments::CSR< TNL::Devices::Host, int >,
                                               TNL::Algorithms::Segments::CSR< TNL::Devices::Host, long >
#ifdef __CUDACC__
                                               ,
                                               TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, int >,
                                               TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, long >
#endif
                                               >;

TYPED_TEST_SUITE( GrowingSegmentsTest, GrowingSegmentsTypes );

template< typename SegmentsType >
void
newSlotTest()
{
   using IndexType = typename SegmentsType::IndexType;
   using DeviceType = typename SegmentsType::DeviceType;
   using GrowingSegmentsType = TNL::Algorithms::Segments::GrowingSegments< SegmentsType >;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GrowingSegmentsType segments( VectorType( 10, 5 ) );
   VectorType data( segments.getStorageSize(), 0 );

   auto data_view = data.getView();
   auto f1 = [ = ] __cuda_callable__( IndexType i ) mutable
   {
      data_view[ segments.newSlot( i ) ] = 1;
   };
   TNL::Algorithms::parallelFor< DeviceType >( 0, segments.getSegmentsCount(), f1 );
   EXPECT_EQ( sum( data ), 10 );
   auto f2 = [ = ] __cuda_callable__( IndexType i ) mutable
   {
      data_view[ segments.newSlot( i ) ] = 2;
   };
   TNL::Algorithms::parallelFor< DeviceType >( 0, segments.getSegmentsCount(), f2 );
   EXPECT_EQ( sum( data ), 30 );
}

TYPED_TEST( GrowingSegmentsTest, newSlot )
{
   using SegmentsType = typename TestFixture::SegmentsType;
   newSlotTest< SegmentsType >();
}

template< typename SegmentsType >
void
deleteSlotTest()
{
   using IndexType = typename SegmentsType::IndexType;
   using DeviceType = typename SegmentsType::DeviceType;
   using GrowingSegmentsType = TNL::Algorithms::Segments::GrowingSegments< SegmentsType >;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GrowingSegmentsType segments( VectorType( 10, 5 ) );
   VectorType data( segments.getStorageSize(), 0 );

   auto data_view = data.getView();
   auto f1 = [ = ] __cuda_callable__( IndexType i ) mutable
   {
      data_view[ segments.newSlot( i ) ] = 1;
      data_view[ segments.newSlot( i ) ] = 2;
   };
   TNL::Algorithms::parallelFor< DeviceType >( 0, segments.getSegmentsCount(), f1 );
   EXPECT_EQ( sum( data ), 30 );

   auto f2 = [ = ] __cuda_callable__( IndexType i ) mutable
   {
      data_view[ segments.deleteSlot( i ) ] = 0;
   };
   TNL::Algorithms::parallelFor< DeviceType >( 0, segments.getSegmentsCount(), f2 );
   EXPECT_EQ( sum( data ), 10 );
}

TYPED_TEST( GrowingSegmentsTest, deleteSlot )
{
   using SegmentsType = typename TestFixture::SegmentsType;
   deleteSlotTest< SegmentsType >();
}

template< typename SegmentsType >
void
forElementsTest()
{
   using IndexType = typename SegmentsType::IndexType;
   using DeviceType = typename SegmentsType::DeviceType;
   using GrowingSegmentsType = TNL::Algorithms::Segments::GrowingSegments< SegmentsType >;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GrowingSegmentsType segments( VectorType( 10, 10 ) );
   VectorType data( segments.getStorageSize(), 0 );

   auto data_view = data.getView();
   auto f1 = [ = ] __cuda_callable__( IndexType i ) mutable
   {
      for( IndexType j = 0; j <= i; j++ )
         data_view[ segments.newSlot( i ) ] = j + 1;
   };
   TNL::Algorithms::parallelFor< DeviceType >( 0, segments.getSegmentsCount(), f1 );

   EXPECT_EQ( sum( data ), 220 );
   auto f2 = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable
   {
      data_view[ globalIdx ] *= -1;
   };
   segments.forAllElements( f2 );
   EXPECT_EQ( sum( data ), -220 );
}

TYPED_TEST( GrowingSegmentsTest, forElements )
{
   using SegmentsType = typename TestFixture::SegmentsType;
   forElementsTest< SegmentsType >();
}

template< typename SegmentsType >
void
reduceSegmentsTest()
{
   using IndexType = typename SegmentsType::IndexType;
   using DeviceType = typename SegmentsType::DeviceType;
   using GrowingSegmentsType = TNL::Algorithms::Segments::GrowingSegments< SegmentsType >;
   using VectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GrowingSegmentsType segments( VectorType( 10, 10 ) );
   VectorType data( segments.getStorageSize(), 0 );

   auto data_view = data.getView();
   auto f1 = [ = ] __cuda_callable__( IndexType i ) mutable
   {
      for( IndexType j = 0; j <= i; j++ )
         data_view[ segments.newSlot( i ) ] = j + 1;
   };
   TNL::Algorithms::parallelFor< DeviceType >( 0, segments.getSegmentsCount(), f1 );
   EXPECT_EQ( sum( data ), 220 );
   data.forAllElements(
      [] __cuda_callable__( IndexType i, IndexType & value ) mutable
      {
         if( value == 0 )
            value = -5;
      } );

   VectorType result( segments.getSegmentsCount(), 0 );
   auto result_view = result.getView();
   auto fetch = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx, bool compute )
   {
      return data_view[ globalIdx ];
   };
   auto keep = [ = ] __cuda_callable__( IndexType segmentIdx, IndexType value ) mutable
   {
      result_view[ segmentIdx ] = value;
   };
   segments.reduceAllSegments( fetch, TNL::Plus{}, keep, (IndexType) 0 );
   EXPECT_EQ( result, VectorType( { 1, 3, 6, 10, 15, 21, 28, 36, 45, 55 } ) );
}

// TODO: restore the test after finishing the implementation of the growing segments
/*TYPED_TEST( GrowingSegmentsTest, reduceSegments )
{
   using SegmentsType = typename TestFixture::SegmentsType;
   reduceSegmentsTest< SegmentsType >();
}*/

#include "../../main.h"
