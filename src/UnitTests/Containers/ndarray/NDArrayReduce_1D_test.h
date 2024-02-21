#pragma once

#include <gtest/gtest.h>
#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/ndarray/Reduce.h>

using namespace TNL;
using namespace TNL::Containers;

template< typename T >
class NDArrayReduce1DTest : public ::testing::Test
{
protected:
   using Array1D = T;
   using IndexType = typename Array1D::IndexType;

   Array1D a;
   typename Array1D::ValueType result;

public:
   NDArrayReduce1DTest()
   {
      init();
   }

   // Due to nvcc limitations, the __cuda_callable__ lambda cannot be directly in the constructor :-(
   void
   init()
   {
      a.setSizes( 10000 );
      auto a_view = a.getView();
      a.forAll(
         [ a_view ] __cuda_callable__( IndexType i ) mutable
         {
            a_view( i ) = i;
         } );
   }
};

using Types = ::testing::Types<  //
   NDArray< int, SizesHolder< int, 0 >, std::index_sequence< 0 >, Devices::Host >
#ifdef __CUDACC__
   ,
   NDArray< int, SizesHolder< int, 0 >, std::index_sequence< 0 >, Devices::Cuda >
#endif
   >;

TYPED_TEST_SUITE( NDArrayReduce1DTest, Types );

template< typename Input, typename Output >
void
test_NDArrayReduce1D( const Input& a, Output result )
{
   const int size = a.template getSize< 0 >();

   // first option to run the nd_reduce function
   nd_reduce( a, std::plus<>{}, 0, result );
   EXPECT_EQ( result, 0.5 * size * ( size - 1 ) );

   // second option to run the nd_reduce function
   result = nd_reduce( a, std::plus<>{}, 0 );
   EXPECT_EQ( result, 0.5 * size * ( size - 1 ) );
}

TYPED_TEST( NDArrayReduce1DTest, scalarProduct )
{
   test_NDArrayReduce1D( this->a, this->result );
}
