#pragma once

#include <gtest/gtest.h>
#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/ndarray/Reduce.h>

using namespace TNL;
using namespace TNL::Containers;

template< typename T >
class NDArrayReduce3DTest : public ::testing::Test
{
protected:
   using Array3D = T;
   using IndexType = typename Array3D::IndexType;
   using Array2D_host = NDArray< typename Array3D::ValueType,  //
                                 SizesHolder< IndexType, 0, 0 >,
                                 std::index_sequence< 0, 1 >,
                                 Devices::Host >;

   Array3D a;
   Array2D_host result_host;

public:
   NDArrayReduce3DTest()
   {
      init();
   }

   // Due to nvcc limitations, the __cuda_callable__ lambda cannot be directly in the constructor :-(
   void
   init()
   {
      a.setSizes( 100, 50, 30 );
      auto a_view = a.getView();
      a.forAll(
         [ a_view ] __cuda_callable__( IndexType i, IndexType j, IndexType k ) mutable
         {
            a_view( i, j, k ) = i;
         } );
   }
};

using Types = ::testing::Types<  //
   NDArray< int, SizesHolder< int, 0, 0, 0 >, std::index_sequence< 0, 1, 2 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0 >, std::index_sequence< 0, 2, 1 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0 >, std::index_sequence< 1, 0, 2 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0 >, std::index_sequence< 1, 2, 0 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0 >, std::index_sequence< 2, 1, 0 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0 >, std::index_sequence< 2, 0, 1 >, Devices::Host >
#ifdef __CUDACC__
   ,
   NDArray< int, SizesHolder< int, 0, 0, 0 >, std::index_sequence< 0, 1, 2 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0 >, std::index_sequence< 0, 2, 1 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0 >, std::index_sequence< 1, 0, 2 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0 >, std::index_sequence< 1, 2, 0 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0 >, std::index_sequence< 2, 1, 0 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0 >, std::index_sequence< 2, 0, 1 >, Devices::Cuda >
#endif
   >;

TYPED_TEST_SUITE( NDArrayReduce3DTest, Types );

template< std::size_t axis = 0, typename Input, typename Output >
void
test_NDArrayReduce3D( const Input& a, Output& result )
{
   //2 options to run nd_reduce function
   nd_reduce< axis >( a, std::plus<>{}, 0, result );
   //result = nd_reduce< axis >( a, std::plus<>{}, 0 );

   constexpr std::size_t axis1 = max( 2 - axis, 1 ) - 1;
   constexpr std::size_t axis2 = min( 3 - axis, 2 );

   EXPECT_EQ( result.template getSize< 0 >(), a.template getSize< axis1 >() );
   EXPECT_EQ( result.template getSize< 1 >(), a.template getSize< axis2 >() );

   if( axis == 0 ) {
      int size = a.template getSize< 0 >();
      for( int i = 0; i < a.template getSize< 1 >(); i++ ) {
         for( int j = 0; j < a.template getSize< 2 >(); j++ ) {
            EXPECT_EQ( result( i, j ), 0.5 * size * ( size - 1 ) );
         }
      }
   }
   else if( axis == 1 ) {
      int size = a.template getSize< 1 >();
      for( int i = 0; i < a.template getSize< 0 >(); i++ ) {
         for( int j = 0; j < a.template getSize< 2 >(); j++ ) {
            EXPECT_EQ( result( i, j ), size * i );
         }
      }
   }
   else if( axis == 2 ) {
      int size = a.template getSize< 2 >();
      for( int i = 0; i < a.template getSize< 0 >(); i++ ) {
         for( int j = 0; j < a.template getSize< 1 >(); j++ ) {
            EXPECT_EQ( result( i, j ), size * i );
         }
      }
   }
}

TYPED_TEST( NDArrayReduce3DTest, Product )
{
   TNL::Algorithms::staticFor< std::size_t, 0, 3 >(
      [ & ]( auto axis )
      {
         test_NDArrayReduce3D< axis >( this->a, this->result_host );
      } );
}
