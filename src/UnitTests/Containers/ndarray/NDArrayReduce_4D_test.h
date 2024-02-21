#pragma once

#include <gtest/gtest.h>
#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/ndarray/Reduce.h>

using namespace TNL;
using namespace TNL::Containers;

template< typename T >
class NDArrayReduce4DTest : public ::testing::Test
{
protected:
   using Array4D = T;
   using IndexType = typename Array4D::IndexType;
   using Array3D = NDArray< typename Array4D::ValueType,  //
                            SizesHolder< IndexType, 0, 0, 0 >,
                            std::index_sequence< 0, 1, 2 >,
                            typename T::DeviceType >;

   using Array3D_host = NDArray< typename Array4D::ValueType,  //
                                 SizesHolder< IndexType, 0, 0, 0 >,
                                 std::index_sequence< 0, 1, 2 >,
                                 Devices::Host >;

   Array4D a;
   Array3D result;
   Array3D_host result_host;

public:
   NDArrayReduce4DTest()
   {
      init();
   }

   // Due to nvcc limitations, the __cuda_callable__ lambda cannot be directly in the constructor :-(
   void
   init()
   {
      a.setSizes( 5, 4, 3, 2 );
      auto a_view = a.getView();
      a.forAll(
         [ a_view ] __cuda_callable__( IndexType i1, IndexType i2, IndexType i3, IndexType i4 ) mutable
         {
            a_view( i1, i2, i3, i4 ) = i1;
         } );
   }
};

using Types = ::testing::Types<  //
   NDArray< int, SizesHolder< int, 0, 0, 0, 0 >, std::index_sequence< 0, 1, 2, 3 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0 >, std::index_sequence< 1, 2, 3, 0 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0 >, std::index_sequence< 2, 3, 1, 0 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0 >, std::index_sequence< 3, 0, 1, 2 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0 >, std::index_sequence< 3, 2, 0, 1 >, Devices::Host >
#ifdef __CUDACC__
   ,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0 >, std::index_sequence< 0, 1, 2, 3 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0 >, std::index_sequence< 1, 2, 3, 0 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0 >, std::index_sequence< 2, 3, 1, 0 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0 >, std::index_sequence< 3, 0, 1, 2 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0 >, std::index_sequence< 3, 2, 0, 1 >, Devices::Cuda >
#endif
   >;

TYPED_TEST_SUITE( NDArrayReduce4DTest, Types );

template< std::size_t axis = 0, typename Input, typename Output, typename Output_host >
void
test_NDArrayReduce4D( const Input& a, Output& result, Output_host& result_host )
{
   //2 options to run nd_reduce function
   nd_reduce< axis >( a, std::plus<>{}, 0, result );
   //result = nd_reduce< axis >( a, std::plus<>{}, 0 );

   constexpr std::size_t axis1 = axis >= 1 ? 0 : 1;
   constexpr std::size_t axis2 = axis >= 2 ? 1 : 2;
   constexpr std::size_t axis3 = axis == 3 ? 2 : 3;

   result_host = result;

   EXPECT_EQ( result.template getSize< 0 >(), a.template getSize< axis1 >() );
   EXPECT_EQ( result.template getSize< 1 >(), a.template getSize< axis2 >() );
   EXPECT_EQ( result.template getSize< 2 >(), a.template getSize< axis3 >() );

   if( axis == 0 ) {
      int size = a.template getSize< 0 >();
      for( int i = 0; i < a.template getSize< 1 >(); i++ ) {
         for( int j = 0; j < a.template getSize< 2 >(); j++ ) {
            for( int k = 0; k < a.template getSize< 3 >(); k++ ) {
               EXPECT_EQ( result_host( i, j, k ), 0.5 * size * ( size - 1 ) );
            }
         }
      }
   }
   else if( axis == 1 ) {
      int size = a.template getSize< 1 >();
      for( int i = 0; i < a.template getSize< 0 >(); i++ ) {
         for( int j = 0; j < a.template getSize< 2 >(); j++ ) {
            for( int k = 0; k < a.template getSize< 3 >(); k++ ) {
               EXPECT_EQ( result_host( i, j, k ), size * i );
            }
         }
      }
   }
   else if( axis == 2 ) {
      int size = a.template getSize< 2 >();
      for( int i = 0; i < a.template getSize< 0 >(); i++ ) {
         for( int j = 0; j < a.template getSize< 1 >(); j++ ) {
            for( int k = 0; k < a.template getSize< 3 >(); k++ ) {
               EXPECT_EQ( result_host( i, j, k ), size * i );
            }
         }
      }
   }
   else if( axis == 3 ) {
      int size = a.template getSize< 3 >();
      for( int i = 0; i < a.template getSize< 0 >(); i++ ) {
         for( int j = 0; j < a.template getSize< 1 >(); j++ ) {
            for( int k = 0; k < a.template getSize< 2 >(); k++ ) {
               EXPECT_EQ( result_host( i, j, k ), size * i );
            }
         }
      }
   }
}

TYPED_TEST( NDArrayReduce4DTest, Reduction )
{
   TNL::Algorithms::staticFor< std::size_t, 0, 4 >(
      [ & ]( auto axis )
      {
         test_NDArrayReduce4D< axis >( this->a, this->result, this->result_host );
      } );
}
