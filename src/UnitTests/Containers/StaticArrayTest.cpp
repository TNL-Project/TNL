#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/StaticVector.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;

static const char* TEST_FILE_NAME = "test_StaticArrayTest.tnl";

// test fixture for typed tests
template< typename Array >
class StaticArrayTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
   using ValueType = typename Array::ValueType;
};

// types for which ArrayTest is instantiated
using StaticArrayTypes = ::testing::Types< StaticArray< 1, short >,
                                           StaticArray< 2, short >,
                                           StaticArray< 3, short >,
                                           StaticArray< 4, short >,
                                           StaticArray< 5, short >,
                                           StaticArray< 1, int >,
                                           StaticArray< 2, int >,
                                           StaticArray< 3, int >,
                                           StaticArray< 4, int >,
                                           StaticArray< 5, int >,
                                           StaticArray< 1, long >,
                                           StaticArray< 2, long >,
                                           StaticArray< 3, long >,
                                           StaticArray< 4, long >,
                                           StaticArray< 5, long >,
                                           StaticArray< 1, float >,
                                           StaticArray< 2, float >,
                                           StaticArray< 3, float >,
                                           StaticArray< 4, float >,
                                           StaticArray< 5, float >,
                                           StaticArray< 1, double >,
                                           StaticArray< 2, double >,
                                           StaticArray< 3, double >,
                                           StaticArray< 4, double >,
                                           StaticArray< 5, double >,
                                           StaticVector< 1, short >,
                                           StaticVector< 1, int >,
                                           StaticVector< 1, long >,
                                           StaticVector< 1, float >,
                                           StaticVector< 1, double >,
                                           StaticVector< 2, short >,
                                           StaticVector< 2, int >,
                                           StaticVector< 2, long >,
                                           StaticVector< 2, float >,
                                           StaticVector< 2, double >,
                                           StaticVector< 3, short >,
                                           StaticVector< 3, int >,
                                           StaticVector< 3, long >,
                                           StaticVector< 3, float >,
                                           StaticVector< 3, double >,
                                           StaticVector< 4, short >,
                                           StaticVector< 4, int >,
                                           StaticVector< 4, long >,
                                           StaticVector< 4, float >,
                                           StaticVector< 4, double >,
                                           StaticVector< 5, short >,
                                           StaticVector< 5, int >,
                                           StaticVector< 5, long >,
                                           StaticVector< 5, float >,
                                           StaticVector< 5, double > >;

TYPED_TEST_SUITE( StaticArrayTest, StaticArrayTypes );

TYPED_TEST( StaticArrayTest, constructors )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ValueType = typename TestFixture::ValueType;
   constexpr int size = ArrayType::getSize();

   ValueType data[ size ];
   for( int i = 0; i < size; i++ )
      data[ i ] = i;

   ArrayType u0;
   EXPECT_TRUE( u0.getData() );

   ArrayType u1( data );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( u1[ i ], data[ i ] );

   ArrayType u2( 7 );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( u2[ i ], 7 );

   ArrayType u3( u1 );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( u3[ i ], u1[ i ] );

   // initialization with 0 requires special treatment to avoid ambiguity,
   // see https://stackoverflow.com/q/4610503
   ArrayType u4( 0 );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( u4[ i ], 0 );

   std::initializer_list< int > il = { 1, 2, 3, 4, 5 };
   ArrayType u5( il );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( u5[ i ], i + 1 );

   std::array< ValueType, size > a;
   for( int i = 0; i < size; i++ )
      a[ i ] = i + 1;
   ArrayType u6( a );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( u6[ i ], i + 1 );
}

TYPED_TEST( StaticArrayTest, getSize )
{
   using ArrayType = typename TestFixture::ArrayType;
   constexpr int size = ArrayType::getSize();

   ArrayType u;
   EXPECT_EQ( u.getSize(), size );
}

TYPED_TEST( StaticArrayTest, getData )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u1;
   EXPECT_TRUE( u1.getData() );

   const ArrayType& u2 = u1;
   EXPECT_TRUE( u2.getData() );
}

template< typename Value >
void
checkCoordinates( StaticArray< 1, Value >& u )
{
   EXPECT_EQ( u.x(), 0 );
   u.x() += 1;
   EXPECT_EQ( u.x(), 1 );
   EXPECT_EQ( u[ 0 ], 1 );
   EXPECT_EQ( u( 0 ), 1 );
}

template< typename Value >
void
checkCoordinates( StaticArray< 2, Value >& u )
{
   EXPECT_EQ( u.x(), 0 );
   EXPECT_EQ( u.y(), 1 );
   u.x() += 1;
   u.y() += 1;
   EXPECT_EQ( u.x(), 1 );
   EXPECT_EQ( u[ 0 ], 1 );
   EXPECT_EQ( u( 0 ), 1 );
   EXPECT_EQ( u.y(), 2 );
   EXPECT_EQ( u[ 1 ], 2 );
   EXPECT_EQ( u( 1 ), 2 );
}

template< typename Value >
void
checkCoordinates( StaticArray< 3, Value >& u )
{
   EXPECT_EQ( u.x(), 0 );
   EXPECT_EQ( u.y(), 1 );
   EXPECT_EQ( u.z(), 2 );
   u.x() += 1;
   u.y() += 1;
   u.z() += 1;
   EXPECT_EQ( u.x(), 1 );
   EXPECT_EQ( u[ 0 ], 1 );
   EXPECT_EQ( u( 0 ), 1 );
   EXPECT_EQ( u.y(), 2 );
   EXPECT_EQ( u[ 1 ], 2 );
   EXPECT_EQ( u( 1 ), 2 );
   EXPECT_EQ( u.z(), 3 );
   EXPECT_EQ( u[ 2 ], 3 );
   EXPECT_EQ( u( 2 ), 3 );
}

template< int _size, typename Value >
void
checkCoordinates( StaticArray< _size, Value >& u )
{}

TYPED_TEST( StaticArrayTest, CoordinatesGetter )
{
   using ArrayType = typename TestFixture::ArrayType;
   constexpr int size = ArrayType::getSize();

   ArrayType u;
   for( int i = 0; i < size; i++ )
      u[ i ] = i;

   checkCoordinates( u );
}

TYPED_TEST( StaticArrayTest, ComparisonOperator )
{
   using ArrayType = typename TestFixture::ArrayType;
   constexpr int size = ArrayType::getSize();

   ArrayType u1, u2, u3;

   for( int i = 0; i < size; i++ ) {
      u1[ i ] = 1;
      u2[ i ] = i;
      u3[ i ] = i;
   }

   EXPECT_TRUE( u1 == u1 );
   EXPECT_TRUE( u1 != u2 );
   EXPECT_TRUE( u2 == u3 );

   // comparison with different type
   StaticArray< size, char > u4( 1 );
   EXPECT_TRUE( u1 == u4 );
   EXPECT_TRUE( u2 != u4 );
   EXPECT_TRUE( u3 != u4 );

   for( int i = 0; i < size; i++ )
      u4[ i ] = i;
   EXPECT_TRUE( u1 != u4 );
   EXPECT_TRUE( u2 == u4 );
   EXPECT_TRUE( u3 == u4 );
}

TYPED_TEST( StaticArrayTest, AssignmentOperator )
{
   using ArrayType = typename TestFixture::ArrayType;
   constexpr int size = ArrayType::getSize();

   ArrayType u1, u2, u3;

   for( int i = 0; i < size; i++ ) {
      u1[ i ] = 1;
      u2[ i ] = i;
   }

   u3 = u1;
   EXPECT_TRUE( u3 == u1 );
   EXPECT_TRUE( u3 != u2 );

   u3 = u2;
   EXPECT_TRUE( u3 == u2 );
   EXPECT_TRUE( u3 != u1 );

   // assignment from different type
   StaticArray< size, char > u4( 127 );
   u3 = u4;
   EXPECT_TRUE( u3 == u4 );

   // assignment of number
   u3 = 0;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( u3[ i ], 0 );
   u3 = 1;
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( u3[ i ], 1 );
}

TYPED_TEST( StaticArrayTest, setValue )
{
   using ArrayType = typename TestFixture::ArrayType;
   constexpr int size = ArrayType::getSize();

   ArrayType u;
   u.setValue( 42 );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( u[ i ], 42 );
}

TYPED_TEST( StaticArrayTest, CastToDifferentStaticArray )
{
   using ArrayType = typename TestFixture::ArrayType;
   constexpr int size = ArrayType::getSize();
   using OtherArray = StaticArray< size, char >;

   ArrayType u1( 1 );
   OtherArray u2( 1 );
   EXPECT_EQ( (OtherArray) u1, u2 );
   EXPECT_EQ( u1, (ArrayType) u2 );
}

TYPED_TEST( StaticArrayTest, SaveAndLoad )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u1( 7 ), u2;
   File file;
   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::out ) );
   ASSERT_NO_THROW( u1.save( file ) );
   ASSERT_NO_THROW( file.close() );
   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::in ) );
   ASSERT_NO_THROW( u2.load( file ) );
   ASSERT_NO_THROW( file.close() );

   EXPECT_EQ( u1, u2 );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );

   ASSERT_NO_THROW( File( TEST_FILE_NAME, std::ios_base::out ) << u1 );
   ASSERT_NO_THROW( File( TEST_FILE_NAME, std::ios_base::in ) >> u2 );

   EXPECT_EQ( u1, u2 );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}

TYPED_TEST( StaticArrayTest, sort )
{
   using ArrayType = typename TestFixture::ArrayType;
   constexpr int size = ArrayType::getSize();

   ArrayType u;
   for( int i = 0; i < size; i++ )
      u[ i ] = size - i - 1;
   u.sort();

   for( int i = 0; i < size; i++ )
      EXPECT_EQ( u[ i ], i );
}

TYPED_TEST( StaticArrayTest, streamOperator )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u = 0;
   std::stringstream testStream;
   testStream << u;
}

template< typename... Ts >
void
unused_variables( const Ts&... )
{}

TYPED_TEST( StaticArrayTest, structuredBinding )
{
   using ArrayType = typename TestFixture::ArrayType;
   constexpr int N = ArrayType::getSize();

   ArrayType array;
   const ArrayType constarray = array;

   if constexpr( N == 1 ) {
      auto& [ lv1 ] = array;
      const auto& [ clv1 ] = constarray;
      auto&& [ rv1 ] = std::move( array );
      const auto&& [ crv1 ] = std::move( constarray );

      // ignore warnings due to unused variables
      unused_variables( lv1, clv1, rv1, crv1 );
   }
   else if constexpr( N == 2 ) {
      auto& [ lv1, lv2 ] = array;
      const auto& [ clv1, clv2 ] = constarray;
      auto&& [ rv1, rv2 ] = std::move( array );
      const auto&& [ crv1, crv2 ] = std::move( constarray );

      // ignore warnings due to unused variables
      unused_variables( lv1, clv1, rv1, crv1 );
      unused_variables( lv2, clv2, rv2, crv2 );
   }
   else if constexpr( N == 3 ) {
      auto& [ lv1, lv2, lv3 ] = array;
      const auto& [ clv1, clv2, clv3 ] = constarray;
      auto&& [ rv1, rv2, rv3 ] = std::move( array );
      const auto&& [ crv1, crv2, crv3 ] = std::move( constarray );

      // ignore warnings due to unused variables
      unused_variables( lv1, clv1, rv1, crv1 );
      unused_variables( lv2, clv2, rv2, crv2 );
      unused_variables( lv3, clv3, rv3, crv3 );
   }
   else if constexpr( N == 4 ) {
      auto& [ lv1, lv2, lv3, lv4 ] = array;
      const auto& [ clv1, clv2, clv3, clv4 ] = constarray;
      auto&& [ rv1, rv2, rv3, rv4 ] = std::move( array );
      const auto&& [ crv1, crv2, crv3, crv4 ] = std::move( constarray );

      // ignore warnings due to unused variables
      unused_variables( lv1, clv1, rv1, crv1 );
      unused_variables( lv2, clv2, rv2, crv2 );
      unused_variables( lv3, clv3, rv3, crv3 );
      unused_variables( lv4, clv4, rv4, crv4 );
   }
   else if constexpr( N == 5 ) {
      auto& [ lv1, lv2, lv3, lv4, lv5 ] = array;
      const auto& [ clv1, clv2, clv3, clv4, clv5 ] = constarray;
      auto&& [ rv1, rv2, rv3, rv4, rv5 ] = std::move( array );
      const auto&& [ crv1, crv2, crv3, crv4, crv5 ] = std::move( constarray );

      // ignore warnings due to unused variables
      unused_variables( lv1, clv1, rv1, crv1 );
      unused_variables( lv2, clv2, rv2, crv2 );
      unused_variables( lv3, clv3, rv3, crv3 );
      unused_variables( lv4, clv4, rv4, crv4 );
      unused_variables( lv5, clv5, rv5, crv5 );
   }
}

#include "../main.h"
