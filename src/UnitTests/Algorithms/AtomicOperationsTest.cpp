// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Devices/Host.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Algorithms;

// Basic CAS: compare matches, value should be swapped
TEST( AtomicOperationsHostCAS, BasicSwapInt )
{
   int address = 42;
   int old = AtomicOperations< Devices::Host >::CAS( address, 42, 100 );
   EXPECT_EQ( old, 42 );
   EXPECT_EQ( address, 100 );
}

// Basic CAS: compare does not match, value should remain unchanged
TEST( AtomicOperationsHostCAS, NoSwapInt )
{
   int address = 42;
   int old = AtomicOperations< Devices::Host >::CAS( address, 99, 100 );
   EXPECT_EQ( old, 42 );
   EXPECT_EQ( address, 42 );
}

// CAS with float: compare matches
TEST( AtomicOperationsHostCAS, BasicSwapFloat )
{
   float address = 3.14f;
   float old = AtomicOperations< Devices::Host >::CAS( address, 3.14f, 2.71f );
   EXPECT_EQ( old, 3.14f );
   EXPECT_EQ( address, 2.71f );
}

// CAS with float: compare does not match
TEST( AtomicOperationsHostCAS, NoSwapFloat )
{
   float address = 3.14f;
   float old = AtomicOperations< Devices::Host >::CAS( address, 0.0f, 2.71f );
   EXPECT_EQ( old, 3.14f );
   EXPECT_EQ( address, 3.14f );
}

// CAS with double: compare matches
TEST( AtomicOperationsHostCAS, BasicSwapDouble )
{
   double address = 1.23456789;
   double old = AtomicOperations< Devices::Host >::CAS( address, 1.23456789, 9.87654321 );
   EXPECT_EQ( old, 1.23456789 );
   EXPECT_EQ( address, 9.87654321 );
}

// CAS with double: compare does not match
TEST( AtomicOperationsHostCAS, NoSwapDouble )
{
   double address = 1.23456789;
   double old = AtomicOperations< Devices::Host >::CAS( address, 0.0, 9.87654321 );
   EXPECT_EQ( old, 1.23456789 );
   EXPECT_EQ( address, 1.23456789 );
}

// CAS swap to zero
TEST( AtomicOperationsHostCAS, SwapToZero )
{
   int address = 7;
   int old = AtomicOperations< Devices::Host >::CAS( address, 7, 0 );
   EXPECT_EQ( old, 7 );
   EXPECT_EQ( address, 0 );
}

// Sequential CAS operations: only the first matching compare should succeed
TEST( AtomicOperationsHostCAS, SequentialCAS )
{
   int address = 10;

   // First CAS: matches, swaps 10 -> 20
   int old1 = AtomicOperations< Devices::Host >::CAS( address, 10, 20 );
   EXPECT_EQ( old1, 10 );
   EXPECT_EQ( address, 20 );

   // Second CAS: compare=10 no longer matches, no swap
   int old2 = AtomicOperations< Devices::Host >::CAS( address, 10, 30 );
   EXPECT_EQ( old2, 20 );
   EXPECT_EQ( address, 20 );

   // Third CAS: compare=20 matches, swaps 20 -> 30
   int old3 = AtomicOperations< Devices::Host >::CAS( address, 20, 30 );
   EXPECT_EQ( old3, 20 );
   EXPECT_EQ( address, 30 );
}

// CAS with negative values
TEST( AtomicOperationsHostCAS, NegativeValues )
{
   int address = -5;
   int old = AtomicOperations< Devices::Host >::CAS( address, -5, -10 );
   EXPECT_EQ( old, -5 );
   EXPECT_EQ( address, -10 );
}

// CAS on already-zero value
TEST( AtomicOperationsHostCAS, ZeroAddress )
{
   int address = 0;
   int old = AtomicOperations< Devices::Host >::CAS( address, 0, 1 );
   EXPECT_EQ( old, 0 );
   EXPECT_EQ( address, 1 );
}

// Parallel CAS: multiple threads attempt to swap; exactly one should succeed
#ifdef HAVE_OPENMP
TEST( AtomicOperationsHostCAS, ParallelAtomicSwap )
{
   int address = 0;
   int successCount = 0;

   #pragma omp parallel for reduction( + : successCount ) num_threads( 8 )
   for( int i = 0; i < 8; i++ ) {
      int old = AtomicOperations< Devices::Host >::CAS( address, 0, 1 );
      if( old == 0 )
         successCount++;
   }

   // Exactly one thread should have seen the original value 0
   EXPECT_EQ( successCount, 1 );
   EXPECT_EQ( address, 1 );
}

// Parallel CAS: increment via CAS loop (spin-based counter)
TEST( AtomicOperationsHostCAS, ParallelCASIncrement )
{
   int counter = 0;
   const int numThreads = 8;
   const int incrementsPerThread = 100;

   #pragma omp parallel num_threads( numThreads )
   {
      for( int i = 0; i < incrementsPerThread; i++ ) {
         int old, expected;
         do {
            expected = counter;
            old = AtomicOperations< Devices::Host >::CAS( counter, expected, expected + 1 );
         } while( old != expected );
      }
   }

   EXPECT_EQ( counter, numThreads * incrementsPerThread );
}
#endif

#include "../main.h"
