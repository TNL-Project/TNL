// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <TNL/Algorithms/Sorting/HostQuicksort.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>

namespace {

template< typename Value, typename Device, typename Index, typename Compare >
void
checkSort( const std::vector< Value >& input, const Compare& compare )
{
   TNL::Containers::Array< Value, Device, Index > array( input );
   TNL::Algorithms::Sorting::HostQuicksort::sort( array, compare );

   std::vector< Value > expected( input );
   std::sort( expected.begin(), expected.end(), compare );

   ASSERT_EQ( static_cast< std::size_t >( array.getSize() ), expected.size() );
   for( std::size_t i = 0; i < expected.size(); ++i )
      ASSERT_EQ( array[ i ], expected[ i ] ) << "mismatch at index " << i;

   // the sorted sequence must not contain any inverted pair
   for( std::size_t i = 1; i < expected.size(); ++i )
      ASSERT_FALSE( compare( array[ i ], array[ i - 1 ] ) ) << "inverted pair at index " << i;
}

template< typename Value, typename Device, typename Index >
void
checkRandomSorts( const std::vector< int >& sizes, std::mt19937& rng )
{
   for( const int size : sizes ) {
      std::vector< Value > values( size );
      std::uniform_int_distribution< int > distribution( -1000000, 1000000 );
      for( Value& value : values )
         value = static_cast< Value >( distribution( rng ) );

      checkSort< Value, Device, Index >( values, std::less<>{} );
      checkSort< Value, Device, Index >( values, std::greater<>{} );
   }
}

}  // namespace

TEST( HostSorterTest, edgeCases )
{
   checkSort< int, TNL::Devices::Host, int >( {}, std::less<>{} );
   checkSort< int, TNL::Devices::Host, int >( { 1 }, std::less<>{} );
   checkSort< int, TNL::Devices::Host, int >( { 2, 1 }, std::less<>{} );
   checkSort< int, TNL::Devices::Host, int >( { 1, 2 }, std::less<>{} );
   checkSort< int, TNL::Devices::Host, int >( { 3, 3, 3, 3, 3 }, std::less<>{} );
}

TEST( HostSorterTest, orderedAndReversed )
{
   std::vector< int > ascending( 5000 );
   std::iota( ascending.begin(), ascending.end(), -2500 );
   checkSort< int, TNL::Devices::Host, int >( ascending, std::less<>{} );

   std::vector< int > descending( ascending.rbegin(), ascending.rend() );
   checkSort< int, TNL::Devices::Host, int >( descending, std::less<>{} );
}

TEST( HostSorterTest, duplicates )
{
   std::vector< int > allEqual( 20000, 42 );
   checkSort< int, TNL::Devices::Host, int >( allEqual, std::less<>{} );

   std::mt19937 rng( 17 );
   std::uniform_int_distribution< int > distribution( 0, 3 );
   std::vector< int > fewDistinct( 100000 );
   for( int& value : fewDistinct )
      value = distribution( rng );
   checkSort< int, TNL::Devices::Host, int >( fewDistinct, std::less<>{} );
}

TEST( HostSorterTest, sizesAroundThresholds )
{
   std::mt19937 rng( 2025 );
   // includes sizes around the 1024-element serial threshold and small/odd sizes
   checkRandomSorts< int, TNL::Devices::Host, int >( { 1, 2, 3, 7, 64, 1023, 1024, 1025, 4096, 65536, 300000 }, rng );
}

TEST( HostSorterTest, valueAndIndexTypes )
{
   std::mt19937 rng( 99 );
   checkRandomSorts< long long, TNL::Devices::Host, int >( { 1000, 100000 }, rng );
   checkRandomSorts< long long, TNL::Devices::Host, std::size_t >( { 100000 }, rng );
   checkRandomSorts< unsigned char, TNL::Devices::Host, int >( { 50000 }, rng );
   checkRandomSorts< float, TNL::Devices::Host, int >( { 100000 }, rng );
   checkRandomSorts< double, TNL::Devices::Host, std::size_t >( { 100000 }, rng );
}

TEST( HostSorterTest, sequentialDevice )
{
   std::mt19937 rng( 1234 );
   checkRandomSorts< int, TNL::Devices::Sequential, int >( { 1, 1025, 100000 }, rng );
}

#include "../../main.h"
