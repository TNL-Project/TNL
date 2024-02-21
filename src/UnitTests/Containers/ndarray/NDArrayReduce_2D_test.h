#pragma once

#include <gtest/gtest.h>
#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/ndarray/Reduce.h>

using namespace TNL;
using namespace TNL::Containers;

template< typename T >
class NDArrayReduce2DTest : public ::testing::Test
{
protected:
   using Array2D = T;
   using IndexType = typename Array2D::IndexType;
   using Array1D = NDArray< typename Array2D::ValueType,  //
                            SizesHolder< IndexType, 0 >,
                            std::index_sequence< 0 >,
                            Devices::Host >;

   Array2D a;
   Array1D result;

public:
   NDArrayReduce2DTest()
   {
      init();
   }

   // Due to nvcc limitations, the __cuda_callable__ lambda cannot be directly in the constructor :-(
   void
   init()
   {
      a.setSizes( 1000, 500 );
      auto a_view = a.getView();
      a.forAll(
         [ a_view ] __cuda_callable__( IndexType i, IndexType j ) mutable
         {
            a_view( i, j ) = i;
         } );
   }
};

using Types = ::testing::Types<  //
   NDArray< int, SizesHolder< int, 0, 0 >, std::index_sequence< 0, 1 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0 >, std::index_sequence< 1, 0 >, Devices::Host >
#ifdef __CUDACC__
   ,
   NDArray< int, SizesHolder< int, 0, 0 >, std::index_sequence< 0, 1 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0 >, std::index_sequence< 1, 0 >, Devices::Cuda >
#endif
   >;

TYPED_TEST_SUITE( NDArrayReduce2DTest, Types );

template< std::size_t axis = 0, typename Input, typename Output >
void
test_NDArrayReduce2D( const Input& a, Output& result )
{
   //2 options to run nd_reduce function
   nd_reduce< axis >( a, std::plus<>{}, 0, result );
   //result = nd_reduce<axis>(a, std::plus<>{}, 0);

   EXPECT_EQ( result.template getSize< 0 >(), a.template getSize< 1 - axis >() );

   if( axis == 0 ) {
      typename Input::IndexType size = a.template getSize< 0 >();
      for( typename Input::IndexType j = 0; j < a.template getSize< 1 >(); j++ ) {
         EXPECT_EQ( result[ j ], 0.5 * size * ( size - 1 ) );
      }
   }
   else {
      typename Input::IndexType size = a.template getSize< 1 >();
      for( typename Input::IndexType j = 0; j < a.template getSize< 0 >(); j++ ) {
         EXPECT_EQ( result[ j ], size * j );
      }
   }
}

TYPED_TEST( NDArrayReduce2DTest, scalarProduct )
{
   TNL::Algorithms::staticFor< std::size_t, 0, 2 >(
      [ & ]( auto axis )
      {
         test_NDArrayReduce2D< axis >( this->a, this->result );
      } );
}
