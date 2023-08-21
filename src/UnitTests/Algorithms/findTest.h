#pragma once

#include <vector>
#include <TNL/Allocators/Host.h>
#include <TNL/Allocators/Cuda.h>
#include <TNL/Algorithms/find.h>
#include <TNL/Containers/Array.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Algorithms;

constexpr int ARRAY_TEST_SIZE = 5000;

// test fixture for typed tests
template< typename Array >
class FindTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
};

// types for which ArrayTest is instantiated
using ContainerTypes = ::testing::Types<
#ifndef __CUDACC__
   Containers::Array< short int, Devices::Sequential >,
   Containers::Array< int, Devices::Sequential >,
   Containers::Array< long int, Devices::Sequential >,
   Containers::Array< float, Devices::Sequential >,
   Containers::Array< double, Devices::Sequential >,
   Containers::Array< short int, Devices::Host >,
   Containers::Array< int, Devices::Host >,
   Containers::Array< long int, Devices::Host >,
   Containers::Array< float, Devices::Host >,
   Containers::Array< double, Devices::Host >
#else
   Containers::Array< short int, Devices::Cuda >,
   Containers::Array< int, Devices::Cuda >,
   Containers::Array< long int, Devices::Cuda >,
   Containers::Array< float, Devices::Cuda >,
   Containers::Array< double, Devices::Cuda >
#endif
   >;

TYPED_TEST_SUITE( FindTest, ContainerTypes );

TYPED_TEST( FindTest, find )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
   EXPECT_EQ( find( u, 0 ).first, true );
   EXPECT_EQ( find( u, 0 ).second, 0 );
   EXPECT_EQ( find( u, 5 ).first, true );
   EXPECT_EQ( find( u, 5 ).second, 5 );
   EXPECT_EQ( find( u, 9 ).first, true );
   EXPECT_EQ( find( u, 9 ).second, 9 );
   EXPECT_EQ( find( u, 10 ).first, false );
}

#include "../main.h"
