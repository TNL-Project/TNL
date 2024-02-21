#pragma once

#include <gtest/gtest.h>
#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/ndarray/Reduce.h>

using namespace TNL;
using namespace TNL::Containers;

template< typename T >
class NDArrayReduce7DTest : public ::testing::Test
{
protected:
   using Array7D = T;
   using IndexType = typename Array7D::IndexType;
   using Array6D = NDArray< typename Array7D::ValueType,
                            SizesHolder< IndexType, 0, 0, 0, 0, 0, 0 >,
                            std::index_sequence< 0, 1, 2, 3, 4, 5 >,
                            typename T::DeviceType >;

   using Array6D_host = NDArray< typename Array7D::ValueType,
                                 SizesHolder< IndexType, 0, 0, 0, 0, 0, 0 >,
                                 std::index_sequence< 0, 1, 2, 3, 4, 5 >,
                                 Devices::Host >;

   Array7D a;
   Array6D result;
   Array6D_host result_host;

public:
   NDArrayReduce7DTest()
   {
      init();
   }

   // Due to nvcc limitations, the __cuda_callable__ lambda cannot be directly in the constructor :-(
   void
   init()
   {
      a.setSizes( 8, 7, 6, 5, 4, 3, 2 );
      auto a_view = a.getView();
      a.forAll(
         [ a_view ] __cuda_callable__(
            IndexType i1, IndexType i2, IndexType i3, IndexType i4, IndexType i5, IndexType i6, IndexType i7 ) mutable
         {
            a_view( i1, i2, i3, i4, i5, i6, i7 ) = i1;
         } );
   }
};

using Types = ::testing::Types<  //
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 0, 1, 2, 3, 4, 5, 6 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 0, 2, 1, 3, 4, 5, 6 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 0, 3, 2, 1, 4, 5, 6 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 1, 0, 3, 2, 4, 5, 6 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 5, 3, 2, 1, 4, 0, 6 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 4, 3, 2, 1, 0, 5, 6 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 2, 3, 0, 1, 4, 5, 6 >, Devices::Host >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 6, 3, 2, 1, 4, 5, 0 >, Devices::Host >
#ifdef __CUDACC__
   ,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 0, 1, 2, 3, 4, 5, 6 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 0, 2, 1, 3, 4, 5, 6 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 0, 3, 2, 1, 4, 5, 6 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 1, 0, 3, 2, 4, 5, 6 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 5, 3, 2, 1, 4, 0, 6 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 4, 3, 2, 1, 0, 5, 6 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 2, 3, 0, 1, 4, 5, 6 >, Devices::Cuda >,
   NDArray< int, SizesHolder< int, 0, 0, 0, 0, 0, 0, 0 >, std::index_sequence< 6, 3, 2, 1, 4, 5, 0 >, Devices::Cuda >
#endif
   >;

TYPED_TEST_SUITE( NDArrayReduce7DTest, Types );

template< std::size_t axis = 0, typename Input, typename Output, typename Output_host >
void
test_NDArrayReduce7D( const Input& a, Output& result, Output_host& result_host )
{
   //2 options to run nd_reduce function
   nd_reduce< axis >( a, std::plus<>{}, 0, result );
   //result = nd_reduce<axis>(a, std::plus<>{}, 0);

   constexpr std::size_t axis1 = axis >= 1 ? 0 : 1;
   constexpr std::size_t axis2 = axis >= 2 ? 1 : 2;
   constexpr std::size_t axis3 = axis >= 3 ? 2 : 3;
   constexpr std::size_t axis4 = axis >= 4 ? 3 : 4;
   constexpr std::size_t axis5 = axis >= 5 ? 4 : 5;
   constexpr std::size_t axis6 = axis >= 6 ? 5 : 6;

   result_host = result;

   EXPECT_EQ( result.template getSize< 0 >(), a.template getSize< axis1 >() );
   EXPECT_EQ( result.template getSize< 1 >(), a.template getSize< axis2 >() );
   EXPECT_EQ( result.template getSize< 2 >(), a.template getSize< axis3 >() );
   EXPECT_EQ( result.template getSize< 3 >(), a.template getSize< axis4 >() );
   EXPECT_EQ( result.template getSize< 4 >(), a.template getSize< axis5 >() );
   EXPECT_EQ( result.template getSize< 5 >(), a.template getSize< axis6 >() );

   if( axis == 0 ) {
      int size = a.template getSize< 0 >();
      for( int i = 0; i < a.template getSize< 1 >(); i++ ) {
         for( int j = 0; j < a.template getSize< 2 >(); j++ ) {
            for( int k = 0; k < a.template getSize< 3 >(); k++ ) {
               for( int v = 0; v < a.template getSize< 4 >(); v++ ) {
                  for( int c = 0; c < a.template getSize< 5 >(); c++ ) {
                     for( int b = 0; b < a.template getSize< 6 >(); b++ ) {
                        EXPECT_EQ( result_host( i, j, k, v, c, b ), 0.5 * size * ( size - 1 ) );
                     }
                  }
               }
            }
         }
      }
   }
   else if( axis == 1 ) {
      int size = a.template getSize< 1 >();
      for( int i = 0; i < a.template getSize< 0 >(); i++ ) {
         for( int j = 0; j < a.template getSize< 2 >(); j++ ) {
            for( int k = 0; k < a.template getSize< 3 >(); k++ ) {
               for( int v = 0; v < a.template getSize< 4 >(); v++ ) {
                  for( int c = 0; c < a.template getSize< 5 >(); c++ ) {
                     for( int b = 0; b < a.template getSize< 6 >(); b++ ) {
                        EXPECT_EQ( result_host( i, j, k, v, c, b ), size * i );
                     }
                  }
               }
            }
         }
      }
   }
   else if( axis == 2 ) {
      int size = a.template getSize< 2 >();
      for( int i = 0; i < a.template getSize< 0 >(); i++ ) {
         for( int j = 0; j < a.template getSize< 1 >(); j++ ) {
            for( int k = 0; k < a.template getSize< 3 >(); k++ ) {
               for( int v = 0; v < a.template getSize< 4 >(); v++ ) {
                  for( int c = 0; c < a.template getSize< 5 >(); c++ ) {
                     for( int b = 0; b < a.template getSize< 6 >(); b++ ) {
                        EXPECT_EQ( result_host( i, j, k, v, c, b ), size * i );
                     }
                  }
               }
            }
         }
      }
   }
   else if( axis == 3 ) {
      int size = a.template getSize< 3 >();
      for( int i = 0; i < a.template getSize< 0 >(); i++ ) {
         for( int j = 0; j < a.template getSize< 1 >(); j++ ) {
            for( int k = 0; k < a.template getSize< 2 >(); k++ ) {
               for( int v = 0; v < a.template getSize< 4 >(); v++ ) {
                  for( int c = 0; c < a.template getSize< 5 >(); c++ ) {
                     for( int b = 0; b < a.template getSize< 6 >(); b++ ) {
                        EXPECT_EQ( result_host( i, j, k, v, c, b ), size * i );
                     }
                  }
               }
            }
         }
      }
   }
   else if( axis == 4 ) {
      int size = a.template getSize< 4 >();
      for( int i = 0; i < a.template getSize< 0 >(); i++ ) {
         for( int j = 0; j < a.template getSize< 1 >(); j++ ) {
            for( int k = 0; k < a.template getSize< 2 >(); k++ ) {
               for( int v = 0; v < a.template getSize< 3 >(); v++ ) {
                  for( int c = 0; c < a.template getSize< 5 >(); c++ ) {
                     for( int b = 0; b < a.template getSize< 6 >(); b++ ) {
                        EXPECT_EQ( result_host( i, j, k, v, c, b ), size * i );
                     }
                  }
               }
            }
         }
      }
   }
   else if( axis == 5 ) {
      int size = a.template getSize< 5 >();
      for( int i = 0; i < a.template getSize< 0 >(); i++ ) {
         for( int j = 0; j < a.template getSize< 1 >(); j++ ) {
            for( int k = 0; k < a.template getSize< 2 >(); k++ ) {
               for( int v = 0; v < a.template getSize< 3 >(); v++ ) {
                  for( int c = 0; c < a.template getSize< 4 >(); c++ ) {
                     for( int b = 0; b < a.template getSize< 6 >(); b++ ) {
                        EXPECT_EQ( result_host( i, j, k, v, c, b ), size * i );
                     }
                  }
               }
            }
         }
      }
   }
   else if( axis == 6 ) {
      int size = a.template getSize< 6 >();
      for( int i = 0; i < a.template getSize< 0 >(); i++ ) {
         for( int j = 0; j < a.template getSize< 1 >(); j++ ) {
            for( int k = 0; k < a.template getSize< 2 >(); k++ ) {
               for( int v = 0; v < a.template getSize< 3 >(); v++ ) {
                  for( int c = 0; c < a.template getSize< 4 >(); c++ ) {
                     for( int b = 0; b < a.template getSize< 5 >(); b++ ) {
                        EXPECT_EQ( result_host( i, j, k, v, c, b ), size * i );
                     }
                  }
               }
            }
         }
      }
   }
}

TYPED_TEST( NDArrayReduce7DTest, Product )
{
   // TODO: general nd_reduce algorithm is not implemented yet
   //test_NDArrayReduce7D< 4 >( this->a, this->result, this->result_host );
   //TNL::Algorithms::staticFor< std::size_t, 0, 7 >(
   //   [ & ]( auto axis )
   //   {
   //      test_NDArrayReduce7D< axis >( this->a, this->result, this->result_host );
   //   } );
}
