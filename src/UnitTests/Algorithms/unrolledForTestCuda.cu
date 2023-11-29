#include <array>

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/unrolledFor.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Algorithms;

template< int N >
void
test_cuda()
{
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   Array a( N );
   a.setValue( 0 );
   auto view = a.getView();

   auto kernel = [ = ] __cuda_callable__( int j ) mutable
   {
      unrolledFor< int, 0, N >(
         [ &view ]( auto i )
         {
            view[ i ] += 1;
         } );
   };
   parallelFor< Devices::Cuda >( 0, 1, kernel );

   ArrayHost expected;
   expected.setSize( N );
   expected.setValue( 1 );

   ArrayHost ah;
   ah = a;
   EXPECT_EQ( ah, expected );
}

TEST( unrolledForTest, cuda_size_8 )
{
   test_cuda< 8 >();
}

TEST( unrolledForTest, cuda_size_97 )
{
   test_cuda< 97 >();
}

TEST( unrolledForTest, cuda_size_5000 )
{
   test_cuda< 5000 >();
}

#include "../main.h"
