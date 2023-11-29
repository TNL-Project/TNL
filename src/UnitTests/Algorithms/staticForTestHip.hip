#include <array>

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/staticFor.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Algorithms;

// nvcc does not allow __cuda_callable__ lambdas inside private regions
void
test_hip_dynamic()
{
   using Array = Containers::Array< int, Devices::Hip >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   constexpr int N = 5;
   Array a( N );
   a.setValue( 0 );
   auto view = a.getView();

   auto kernel = [ = ] __cuda_callable__( int j ) mutable
   {
      staticFor< int, 0, N >(
         [ &view ]( auto i )
         {
            view[ i ] += 1;
         } );
   };
   parallelFor< Devices::Hip >( 0, 1, kernel );

   ArrayHost expected;
   expected.setSize( N );
   expected.setValue( 1 );

   ArrayHost ah;
   ah = a;
   EXPECT_EQ( ah, expected );
}

TEST( staticForTest, hip_dynamic )
{
   test_hip_dynamic();
}

template< int i, typename View >
__cuda_callable__
void
static_helper( View& view )
{
   view[ i ] += 1;
}

// nvcc does not allow __cuda_callable__ lambdas inside private regions
void
test_hip_static()
{
   using Array = Containers::Array< int, Devices::Hip >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   constexpr int N = 5;
   Array a( N );
   a.setValue( 0 );
   auto view = a.getView();

   auto kernel = [ = ] __cuda_callable__( int j ) mutable
   {
      staticFor< int, 0, N >(
         [ &view ]( auto i )
         {
            static_helper< i >( view );
         } );
   };
   parallelFor< Devices::Hip >( 0, 1, kernel );

   ArrayHost expected;
   expected.setSize( N );
   expected.setValue( 1 );

   ArrayHost ah;
   ah = a;
   EXPECT_EQ( ah, expected );
}

TEST( staticForTest, hip_static )
{
   test_hip_static();
}

#include "../main.h"
