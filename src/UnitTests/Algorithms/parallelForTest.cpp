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

#include "../main.h"
