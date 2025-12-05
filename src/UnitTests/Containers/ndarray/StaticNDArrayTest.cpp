#include "gtest/gtest.h"

#include <TNL/Containers/NDArray.h>

using namespace TNL::Containers;
using std::index_sequence;

template< typename ArrayView >
void
expect_identity( const ArrayView& a )
{
   Array< typename ArrayView::ValueType, typename ArrayView::DeviceType, typename ArrayView::IndexType > identity;
   identity.setLike( a );
   for( int i = 0; i < identity.getSize(); i++ )
      identity[ i ] = i;
   EXPECT_EQ( a, identity );
}

TEST( StaticNDArrayTest, Static_2D_Identity )
{
   constexpr int I = 3;
   constexpr int J = 5;
   StaticNDArray< int, SizesHolder< int, I, J > > a;

   int v = 0;
   for( int i = 0; i < I; i++ )
      for( int j = 0; j < J; j++ )
         a( i, j ) = v++;

   expect_identity( a.getStorageArrayView() );
}

TEST( StaticNDArrayTest, Static_2D_Permuted )
{
   constexpr int I = 3;
   constexpr int J = 5;
   StaticNDArray< int, SizesHolder< int, I, J >, index_sequence< 1, 0 > > a;

   int v = 0;
   for( int j = 0; j < J; j++ )
      for( int i = 0; i < I; i++ )
         a( i, j ) = v++;

   expect_identity( a.getStorageArrayView() );
}

TEST( StaticNDArrayTest, Static_6D_Permuted )
{
   constexpr int I = 2;
   constexpr int J = 2;
   constexpr int K = 2;
   constexpr int L = 2;
   constexpr int M = 2;
   constexpr int N = 2;
   StaticNDArray< int, SizesHolder< int, I, J, K, L, M, N >, index_sequence< 5, 3, 4, 2, 0, 1 > > a;

   int v = 0;
   for( int n = 0; n < N; n++ )
      for( int l = 0; l < L; l++ )
         for( int m = 0; m < M; m++ )
            for( int k = 0; k < K; k++ )
               for( int i = 0; i < I; i++ )
                  for( int j = 0; j < J; j++ )
                     a( i, j, k, l, m, n ) = v++;

   expect_identity( a.getStorageArrayView() );
}

TEST( StaticNDArrayTest, CopySemantics )
{
   constexpr int I = 3;
   constexpr int J = 5;
   StaticNDArray< int, SizesHolder< int, I, J > > a;
   StaticNDArray< int, SizesHolder< int, I, J > > b;
   StaticNDArray< int, SizesHolder< int, I, J > > c;

   int v = 0;
   for( int i = 0; i < I; i++ )
      for( int j = 0; j < J; j++ )
         a( i, j ) = v++;

   expect_identity( a.getStorageArrayView() );

   b = a;
   EXPECT_EQ( a, b );

   auto a_view = a.getView();
   auto b_view = b.getView();
   EXPECT_EQ( a_view, b_view );
   EXPECT_EQ( a_view.getView(), b_view );
   EXPECT_EQ( a_view.getConstView(), b_view.getConstView() );
   EXPECT_EQ( a.getConstView(), b.getConstView() );
   EXPECT_EQ( a.getConstView(), b_view.getConstView() );

   auto c_view = c.getView();
   c_view = b_view;
   EXPECT_EQ( a_view, c_view );
   EXPECT_EQ( a_view.getView(), c_view );
   EXPECT_EQ( a_view.getConstView(), c_view.getConstView() );
   EXPECT_EQ( a.getConstView(), c.getConstView() );
   EXPECT_EQ( a.getConstView(), c_view.getConstView() );
}

#include "../../main.h"
