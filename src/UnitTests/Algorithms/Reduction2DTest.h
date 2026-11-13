#pragma once

#include <type_traits>

#include "gtest/gtest.h"

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/Reduction2D.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename View >
void
setLinearSequence( View& deviceVector )
{
   using HostVector = Containers::Vector< typename View::RealType, Devices::Host, typename View::IndexType >;
   HostVector a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = i;
   deviceVector = a;
}

template< typename View >
void
setNegativeLinearSequence( View& deviceVector )
{
   using HostVector = Containers::Vector< typename View::RealType, Devices::Host, typename View::IndexType >;
   HostVector a;
   a.setLike( deviceVector );
   for( int i = 0; i < a.getSize(); i++ )
      a[ i ] = -i;
   deviceVector = a;
}

// test fixture for typed tests
template< typename Vector >
class Reduction2DTest : public ::testing::Test
{
protected:
   using DeviceVector = Vector;
   using DeviceView = VectorView< typename Vector::RealType, typename Vector::DeviceType, typename Vector::IndexType >;
   using HostVector = typename DeviceVector::template Self< typename DeviceVector::RealType, Devices::Sequential >;
   using HostView = typename DeviceView::template Self< typename DeviceView::RealType, Devices::Sequential >;

   // should be small enough to have fast tests, but larger than minGPUReductionDataSize
   // and large enough to require multiple CUDA blocks for reduction
   static constexpr int size = 5000;

   // number of vectors which are reduced together
   static constexpr int n = 4;

   DeviceVector V;
   DeviceVector y;
   HostVector result;

   Reduction2DTest()
   {
      V.setSize( size * n );
      y.setSize( size );
      result.setSize( n );

      for( int i = 0; i < n; i++ ) {
         DeviceView v( V.getData() + i * size, size );
         if( i % 2 == 0 )
            setLinearSequence( v );
         else
            setNegativeLinearSequence( v );
      }
      y.setValue( 1 );
   }
};

// types for which Reduction2DTest is instantiated
using VectorTypes = ::testing::Types<  //
   Vector< int, Devices::Host >,
   Vector< float, Devices::Host >,
   Vector< int, Devices::Sequential >,
   Vector< float, Devices::Sequential >
#if defined( __CUDACC__ )
   ,
   Vector< int, Devices::Cuda >,
   Vector< float, Devices::Cuda >
#endif
#if defined( __HIP__ )
   ,
   Vector< int, Devices::Hip >,
   Vector< float, Devices::Hip >
#endif
   >;

TYPED_TEST_SUITE( Reduction2DTest, VectorTypes );

// idiot nvcc does not allow __cuda_callable__ lambdas inside private or protected regions
template< typename DeviceVector, typename HostVector >
void
test_Reduction2D( const DeviceVector& V, const DeviceVector& y, HostVector& result )
{
   using RealType = typename DeviceVector::RealType;
   using DeviceType = typename DeviceVector::DeviceType;
   using IndexType = typename DeviceVector::IndexType;

   const RealType* _V = V.getData();
   const RealType* _y = y.getData();
   const IndexType size = y.getSize();
   const int n = result.getSize();
   ASSERT_EQ( V.getSize(), size * n );

   auto fetch = [ = ] __cuda_callable__( IndexType i, int k )
   {
      TNL_ASSERT_LT( i, size, "fetcher got invalid index i" );
      TNL_ASSERT_LT( k, n, "fetcher got invalid index k" );
      return _V[ i + k * size ] * _y[ i ];
   };
   Reduction2D< DeviceType >::reduce( (RealType) 0, fetch, std::plus<>{}, size, n, result.getView() );

   for( int i = 0; i < n; i++ ) {
      if( i % 2 == 0 )
         EXPECT_EQ( result[ i ], 0.5 * size * ( size - 1 ) );
      else
         EXPECT_EQ( result[ i ], -0.5 * size * ( size - 1 ) );
   }
}
TYPED_TEST( Reduction2DTest, scalarProduct )
{
   test_Reduction2D( this->V, this->y, this->result );
}

template< typename Device, typename Index >
void
testReduction2DIndexType( Index size, int n )
{
   using Result = long long;
   Vector< Result, Devices::Host, int > result( n );
   auto fetch = [ = ] __cuda_callable__( Index i, int k ) -> Result
   {
      if constexpr( std::is_signed_v< Index > )
         TNL_ASSERT_GE( i, 0, "fetcher got negative index" );
      TNL_ASSERT_LT( i, size, "fetcher got invalid index i" );
      TNL_ASSERT_LT( k, n, "fetcher got invalid index k" );
      return static_cast< Result >( i );
   };
   Reduction2D< Device >::reduce( Result( 0 ), fetch, std::plus<>{}, size, n, result.getView() );
   const Result expected = static_cast< Result >( size ) * ( static_cast< Result >( size ) - 1 ) / 2;
   for( int k = 0; k < n; k++ )
      EXPECT_EQ( result[ k ], expected );
}

template< typename Device >
void
testReduction2DIndexTypes()
{
   testReduction2DIndexType< Device >( static_cast< short >( 1 ), 3 );
   testReduction2DIndexType< Device >( static_cast< short >( 255 ), 3 );
   testReduction2DIndexType< Device >( static_cast< short >( 256 ), 3 );
   testReduction2DIndexType< Device >( static_cast< short >( 257 ), 3 );
   testReduction2DIndexType< Device >( static_cast< short >( 1000 ), 3 );
   testReduction2DIndexType< Device >( static_cast< short >( 32767 ), 3 );
   testReduction2DIndexType< Device >( static_cast< unsigned int >( 1 ), 3 );
   testReduction2DIndexType< Device >( static_cast< unsigned int >( 257 ), 3 );
}

TEST( Reduction2DIndexTypes, Sequential )
{
   testReduction2DIndexTypes< Devices::Sequential >();
}

TEST( Reduction2DIndexTypes, Host )
{
   testReduction2DIndexTypes< Devices::Host >();
}

#if defined( __CUDACC__ )
TEST( Reduction2DIndexTypes, Cuda )
{
   testReduction2DIndexTypes< Devices::Cuda >();
}
#endif

#include "../main.h"
