#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Algorithms/parallelFor.h>

#include <gtest/gtest.h>

using namespace TNL;

TEST( ParallelForTest, 1D_host )
{
   using Array = Containers::Array< int, Devices::Host >;

   Array a;

   for( int size = 1; size <= 100000; size *= 10 ) {
      Array expected;
      expected.setSize( size );
      for( int i = 0; i < size; i++ )
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel = [ = ]( int i ) mutable
      {
         view[ i ] = i;
      };
      Algorithms::parallelFor< Devices::Host >( 0, size, kernel );

      if( a != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( a[ i ], i ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForTest, 2D_host )
{
   using Array = Containers::Array< int, Devices::Host >;
   using MultiIndex = Containers::StaticArray< 2, int >;

   Array a;

   for( int size = 1; size <= 100000; size *= 10 ) {
      Array expected;
      expected.setSize( size );
      for( int i = 0; i < size; i++ )
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel1 = [ = ]( const MultiIndex& i ) mutable
      {
         view[ i.x() ] = i.x();
      };
      Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0 }, MultiIndex{ size, 1 }, kernel1 );

      if( a != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( a[ i ], i ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel2 = [ = ]( const MultiIndex& i ) mutable
      {
         view[ i.y() ] = i.y();
      };
      Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0 }, MultiIndex{ 1, size }, kernel2 );

      if( a != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( a[ i ], i ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForTest, 3D_host )
{
   using Array = Containers::Array< int, Devices::Host >;
   using MultiIndex = Containers::StaticArray< 3, int >;

   Array a;

   for( int size = 1; size <= 100000; size *= 10 ) {
      Array expected;
      expected.setSize( size );
      for( int i = 0; i < size; i++ )
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel1 = [ = ]( const MultiIndex& i ) mutable
      {
         view[ i.x() ] = i.x();
      };
      Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0, 0 }, MultiIndex{ size, 1, 1 }, kernel1 );

      if( a != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( a[ i ], i ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel2 = [ = ]( const MultiIndex& i ) mutable
      {
         view[ i.y() ] = i.y();
      };
      Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, size, 1 }, kernel2 );

      if( a != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( a[ i ], i ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel3 = [ = ]( const MultiIndex& i ) mutable
      {
         view[ i.z() ] = i.z();
      };
      Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, 1, size }, kernel3 );

      if( a != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( a[ i ], i ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForTest, 1D_empty_range_host )
{
   using Array = Containers::Array< int, Devices::Host >;

   Array a( 1 );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ]( int i ) mutable
   {
      view[ 0 ] = 1;
   };

   Algorithms::parallelFor< Devices::Host >( 0, 0, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( 5, 3, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );
}

TEST( ParallelForTest, 2D_empty_range_host )
{
   using MultiIndex = Containers::StaticArray< 2, int >;
   using Array = Containers::Array< int, Devices::Host >;

   Array a( 1 );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ]( const MultiIndex& i ) mutable
   {
      view[ 0 ] = 1;
   };

   // single empty dimension
   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0 }, MultiIndex{ 0, 1 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0 }, MultiIndex{ 1, 0 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   // single reversed dimension
   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 5, 0 }, MultiIndex{ 3, 1 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 5 }, MultiIndex{ 1, 3 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   // both dimensions empty
   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0 }, MultiIndex{ 0, 0 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   // both dimensions reversed
   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 5, 5 }, MultiIndex{ 3, 3 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   // one empty, one reversed
   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 5 }, MultiIndex{ 0, 3 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 5, 0 }, MultiIndex{ 3, 0 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );
}

TEST( ParallelForTest, 3D_empty_range_host )
{
   using MultiIndex = Containers::StaticArray< 3, int >;
   using Array = Containers::Array< int, Devices::Host >;

   Array a( 1 );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ]( const MultiIndex& i ) mutable
   {
      view[ 0 ] = 1;
   };

   // single empty dimension
   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 0, 1, 1 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, 0, 1 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, 1, 0 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   // single reversed dimension
   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0, 5 }, MultiIndex{ 1, 1, 3 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 5, 0 }, MultiIndex{ 1, 3, 1 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 5, 0, 0 }, MultiIndex{ 3, 1, 1 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   // two empty dimensions
   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 0, 0, 1 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 0, 1, 0 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, 0, 0 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   // two reversed dimensions
   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 5, 5 }, MultiIndex{ 10, 3, 3 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 5, 0, 5 }, MultiIndex{ 3, 10, 3 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 5, 5, 0 }, MultiIndex{ 3, 3, 10 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   // all dimensions empty
   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 0, 0, 0 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   // all dimensions reversed
   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 5, 5, 5 }, MultiIndex{ 3, 3, 3 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   // mixed: one valid, one empty, one reversed
   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0, 5 }, MultiIndex{ 10, 0, 3 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 5, 0 }, MultiIndex{ 10, 3, 0 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 5, 0, 0 }, MultiIndex{ 3, 10, 0 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 5, 0, 0 }, MultiIndex{ 3, 0, 10 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 5, 0 }, MultiIndex{ 0, 3, 10 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0, 5 }, MultiIndex{ 0, 10, 3 }, kernel );
   EXPECT_EQ( a.getElement( 0 ), 0 );
}

TEST( ParallelForTest, 1D_above_32bit_range_host )
{
   GTEST_SKIP() << "iterating over more than 2^32 indices would be too slow on the host";

   // see the CUDA variant of this test for the documentation
   constexpr std::size_t size = ( 1UL << 32 ) + 1024;
   constexpr std::size_t tail = 1024;
   constexpr std::size_t offset = size - tail;  // == 2^32

   using Array = Containers::Array< int, Devices::Host, std::size_t >;

   Array a( tail );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ]( std::size_t i ) mutable
   {
      if( i >= offset )
         view[ i - offset ] += 1;
   };
   Algorithms::parallelFor< Devices::Host >( 0, size, kernel );

   for( std::size_t j = 0; j < tail; j++ )
      ASSERT_EQ( a[ j ], 1 ) << "The iteration i = 2^32 + " << j << " was not processed exactly once";
}

TEST( ParallelForTest, 2D_above_32bit_range_host )
{
   GTEST_SKIP() << "iterating over more than 2^32 indices would be too slow on the host";

   // see the CUDA variant of this test for the documentation
   constexpr std::size_t size = ( 1UL << 32 ) + ( 1UL << 31 );
   constexpr std::size_t tail = 1024;
   constexpr std::size_t offset = size - tail;

   using Array = Containers::Array< int, Devices::Host, std::size_t >;
   using MultiIndex = Containers::StaticArray< 2, std::size_t >;

   Array a( tail );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ]( const MultiIndex& i ) mutable
   {
      if( i.x() >= offset )
         view[ i.x() - offset ] += 1;
   };
   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0 }, MultiIndex{ size, 1 }, kernel );

   for( std::size_t j = 0; j < tail; j++ )
      ASSERT_EQ( a[ j ], 1 ) << "The 2D iteration with i.x() = " << offset + j << " was not processed exactly once";
}

TEST( ParallelForTest, 3D_above_32bit_range_host )
{
   GTEST_SKIP() << "iterating over more than 2^32 indices would be too slow on the host";

   // see the CUDA variant of this test for the documentation
   constexpr std::size_t size = ( 1UL << 32 ) + ( 1UL << 31 );
   constexpr std::size_t tail = 1024;
   constexpr std::size_t offset = size - tail;

   using Array = Containers::Array< int, Devices::Host, std::size_t >;
   using MultiIndex = Containers::StaticArray< 3, std::size_t >;

   Array a( tail );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ]( const MultiIndex& i ) mutable
   {
      if( i.x() >= offset )
         view[ i.x() - offset ] += 1;
   };
   Algorithms::parallelFor< Devices::Host >( MultiIndex{ 0, 0, 0 }, MultiIndex{ size, 1, 1 }, kernel );

   for( std::size_t j = 0; j < tail; j++ )
      ASSERT_EQ( a[ j ], 1 ) << "The 3D iteration with i.x() = " << offset + j << " was not processed exactly once";
}

#include "../main.h"
