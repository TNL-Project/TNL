#pragma once

#include "gtest/gtest.h"

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/Reduction3D.h>

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

template< typename Vector >
class Reduction3DTest : public ::testing::Test
{
protected:
   using DeviceVector = Vector;
   using DeviceView = VectorView< typename Vector::RealType, typename Vector::DeviceType, typename Vector::IndexType >;
   using HostVector = typename DeviceVector::template Self< typename DeviceVector::RealType, Devices::Sequential >;
   using HostView = typename DeviceView::template Self< typename DeviceView::RealType, Devices::Sequential >;

   static constexpr int size = 500;
   static constexpr int m = 200;
   static constexpr int n = 100;

   DeviceVector v;
   DeviceVector y;
   HostVector result;

   Reduction3DTest()
   {
      v.setSize( m * n * size );
      y.setSize( size );
      result.setSize( m * n );

      for( int i = 0; i < m; i++ ) {
         for( int j = 0; j < n; j++ ) {
            DeviceView vec( v.getData() + i * n * size + j * size, size );
            setLinearSequence( vec );
         }
      }
      y.setValue( 1 );
   }
};

using VectorTypes = ::testing::Types<  //
   Vector< int, Devices::Sequential >,
   Vector< float, Devices::Sequential >,
   Vector< int, Devices::Host >,
   Vector< float, Devices::Host >
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

TYPED_TEST_SUITE( Reduction3DTest, VectorTypes );

template< typename DeviceVector, typename HostVector >
void
test_Reduction3D( const DeviceVector& V, const DeviceVector& y, HostVector& result, int m, int n )
{
   using RealType = typename DeviceVector::RealType;
   using DeviceType = typename DeviceVector::DeviceType;
   using IndexType = typename DeviceVector::IndexType;

   const RealType* _V = V.getData();
   const RealType* _y = y.getData();
   const IndexType size = y.getSize();

   ASSERT_EQ( V.getSize(), size * m * n );
   ASSERT_EQ( result.getSize(), m * n );

   auto fetch = [ = ] __cuda_callable__( IndexType i, int k, int l )
   {
      TNL_ASSERT_LT( i, size, "fetcher got invalid index i" );
      TNL_ASSERT_LT( k, m, "fetcher got invalid index k" );
      TNL_ASSERT_LT( l, n, "fetcher got invalid index l" );
      return _V[ i + k * size * n + l * size ] * _y[ i ];
   };

   auto result_view = result.getView();

   auto output = [ = ] __cuda_callable__( IndexType i, IndexType j ) mutable -> typename DeviceVector::RealType&
   {
      return result_view( i * n + j );
   };

   Reduction3D< DeviceType >::reduce( (RealType) 0, fetch, std::plus<>{}, size, m, n, output );

   for( int i = 0; i < m; i++ ) {
      for( int j = 0; j < n; j++ ) {
         EXPECT_EQ( result[ i * n + j ], 0.5 * size * ( size - 1 ) );
      }
   }
}

TYPED_TEST( Reduction3DTest, sum )
{
   test_Reduction3D( this->v, this->y, this->result, this->m, this->n );
}

#include "../main.h"
