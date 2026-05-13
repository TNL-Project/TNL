#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Algorithms/parallelFor.h>

#include <gtest/gtest.h>

using namespace TNL;

TEST( ParallelForTest, 1D_hip )
{
   using Array = Containers::Array< int, Devices::Hip >;
   using ArrayHost = Containers::Array< int, Devices::Host >;

   Array a;

   for( int size = 1; size <= 100000000; size *= 100 ) {
      ArrayHost expected;
      expected.setSize( size );
      for( int i = 0; i < size; i++ )
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel = [ = ] __cuda_callable__( int i ) mutable
      {
         view[ i ] = i;
      };
      Algorithms::parallelFor< Devices::Hip >( 0, size, kernel );

      ArrayHost ah;
      ah = a;
      if( ah != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( ah[ i ], expected[ i ] ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForTest, 2D_hip )
{
   using Array = Containers::Array< int, Devices::Hip >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   using MultiIndex = Containers::StaticArray< 2, int >;

   Array a;

   for( int size = 1; size <= 100000000; size *= 100 ) {
      ArrayHost expected;
      expected.setSize( size );
      for( int i = 0; i < size; i++ )
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel1 = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
      {
         view[ i.x() ] = i.x();
      };
      Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0 }, MultiIndex{ size, 1 }, kernel1 );

      ArrayHost ah;
      ah = a;
      if( ah != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( ah[ i ], expected[ i ] ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel2 = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
      {
         view[ i.y() ] = i.y();
      };
      Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0 }, MultiIndex{ 1, size }, kernel2 );

      ah = a;
      if( ah != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( ah[ i ], expected[ i ] ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForTest, 3D_hip )
{
   using Array = Containers::Array< int, Devices::Hip >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   using MultiIndex = Containers::StaticArray< 3, int >;

   Array a;

   for( int size = 1; size <= 100000000; size *= 100 ) {
      ArrayHost expected;
      expected.setSize( size );
      for( int i = 0; i < size; i++ )
         expected[ i ] = i;

      a.setSize( size );
      a.setValue( 0 );
      auto view = a.getView();
      auto kernel1 = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
      {
         view[ i.x() ] = i.x();
      };
      Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0, 0 }, MultiIndex{ size, 1, 1 }, kernel1 );

      ArrayHost ah;
      ah = a;
      if( ah != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( ah[ i ], expected[ i ] ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel2 = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
      {
         view[ i.y() ] = i.y();
      };
      Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, size, 1 }, kernel2 );

      ah = a;
      if( ah != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( ah[ i ], expected[ i ] ) << "First index at which the result is wrong is i = " << i;
      }

      a.setValue( 0 );
      auto kernel3 = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
      {
         view[ i.z() ] = i.z();
      };
      Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, 1, size }, kernel3 );

      ah = a;
      if( ah != expected ) {
         for( int i = 0; i < size; i++ )
            ASSERT_EQ( ah[ i ], expected[ i ] ) << "First index at which the result is wrong is i = " << i;
      }
   }
}

TEST( ParallelForTest, 1D_empty_range_hip )
{
   using Array = Containers::Array< int, Devices::Hip >;
   using ArrayHost = Containers::Array< int, Devices::Host >;

   Array a( 1 );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ] __cuda_callable__( int i ) mutable
   {
      view[ 0 ] = 1;
   };

   ArrayHost ah;

   Algorithms::parallelFor< Devices::Hip >( 0, 0, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( 5, 3, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );
}

TEST( ParallelForTest, 2D_empty_range_hip )
{
   using MultiIndex = Containers::StaticArray< 2, int >;
   using Array = Containers::Array< int, Devices::Hip >;
   using ArrayHost = Containers::Array< int, Devices::Host >;

   Array a( 1 );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
   {
      view[ 0 ] = 1;
   };

   ArrayHost ah;

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0 }, MultiIndex{ 0, 1 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0 }, MultiIndex{ 1, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 5, 0 }, MultiIndex{ 3, 1 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 5 }, MultiIndex{ 1, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0 }, MultiIndex{ 0, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 5, 5 }, MultiIndex{ 3, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 5 }, MultiIndex{ 0, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 5, 0 }, MultiIndex{ 3, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );
}

TEST( ParallelForTest, 3D_empty_range_hip )
{
   using MultiIndex = Containers::StaticArray< 3, int >;
   using Array = Containers::Array< int, Devices::Hip >;
   using ArrayHost = Containers::Array< int, Devices::Host >;

   Array a( 1 );
   a.setValue( 0 );
   auto view = a.getView();
   auto kernel = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
   {
      view[ 0 ] = 1;
   };

   ArrayHost ah;

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 0, 1, 1 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, 0, 1 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, 1, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0, 5 }, MultiIndex{ 1, 1, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 5, 0 }, MultiIndex{ 1, 3, 1 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 5, 0, 0 }, MultiIndex{ 3, 1, 1 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 0, 0, 1 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 0, 1, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 1, 0, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 5, 5 }, MultiIndex{ 10, 3, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 5, 0, 5 }, MultiIndex{ 3, 10, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 5, 5, 0 }, MultiIndex{ 3, 3, 10 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0, 0 }, MultiIndex{ 0, 0, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 5, 5, 5 }, MultiIndex{ 3, 3, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0, 5 }, MultiIndex{ 10, 0, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 5, 0 }, MultiIndex{ 10, 3, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 5, 0, 0 }, MultiIndex{ 3, 10, 0 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 5, 0, 0 }, MultiIndex{ 3, 0, 10 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 5, 0 }, MultiIndex{ 0, 3, 10 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );

   Algorithms::parallelFor< Devices::Hip >( MultiIndex{ 0, 0, 5 }, MultiIndex{ 0, 10, 3 }, kernel );
   ah = a;
   EXPECT_EQ( ah.getElement( 0 ), 0 );
}

#include "../main.h"
