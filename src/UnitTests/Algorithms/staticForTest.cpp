#include <array>

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/staticFor.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Algorithms;

TEST( staticForTest, host_dynamic )
{
   constexpr int N = 5;
   std::array< int, N > a;
   a.fill( 0 );

   staticFor< int, 0, N >(
      [ &a ]( auto i )
      {
         a[ i ] += 1;
      } );

   std::array< int, N > expected;
   expected.fill( 1 );
   EXPECT_EQ( a, expected );
}

TEST( staticForTest, host_static )
{
   constexpr int N = 5;
   std::array< int, N > a;
   a.fill( 0 );

   staticFor< int, 0, N >(
      [ &a ]( auto i )
      {
         std::get< i >( a ) += 1;
      } );

   std::array< int, N > expected;
   expected.fill( 1 );
   EXPECT_EQ( a, expected );
}

TEST( staticForTest, host_empty )
{
   bool called = false;

   staticFor< int, 0, 0 >(
      [ &called ]( auto i )
      {
         called = true;
      } );
   EXPECT_FALSE( called );

   staticFor< int, 0, -1 >(
      [ &called ]( auto i )
      {
         called = true;
      } );
   EXPECT_FALSE( called );
}

#include "../main.h"
