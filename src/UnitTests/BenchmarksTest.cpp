#include <chrono>
#include <cmath>
#include <thread>

#include <gtest/gtest.h>

#include <TNL/Benchmarks/BenchmarkResult.h>
#include <TNL/Benchmarks/Utils.h>

using namespace TNL;
using namespace TNL::Benchmarks;

// Runs timeFunction on the host without the minimal warmup time.
template< typename ComputeFunction >
BenchmarkResult
runTimeFunction( ComputeFunction compute, std::size_t loops, double minTime, std::size_t warmupLoops, bool forceWarmupLoop )
{
   Solvers::IterativeSolverMonitor< double > monitor;
   BenchmarkResult result;
   auto reset = []() {};
   timeFunction< Devices::Host >( compute, reset, loops, minTime, warmupLoops, 0.0, forceWarmupLoop, monitor, result );
   return result;
}

TEST( BenchmarksTest, loops )
{
   int calls = 0;
   auto compute = [ & ]()
   {
      calls++;
   };
   const BenchmarkResult result = runTimeFunction(
      compute,
      5,    // loops
      0.0,  // minTime (s)
      1,    // warmupLoops
      true );  // forceWarmupLoop
   EXPECT_EQ( result.loops, 5 );
   // one warmup iteration
   EXPECT_EQ( calls, 6 );
}

TEST( BenchmarksTest, zeroLoopsRunsOneIteration )
{
   int calls = 0;
   auto compute = [ & ]()
   {
      calls++;
   };
   // no lower bound and no warmup, but at least one sample must be measured
   const BenchmarkResult result = runTimeFunction(
      compute,
      0,    // loops
      0.0,  // minTime (s)
      0,    // warmupLoops, 0 for no warmup
      true );  // forceWarmupLoop
   EXPECT_EQ( result.loops, 1 );
   EXPECT_EQ( calls, 1 );
   EXPECT_GE( result.time, 0.0 );
   EXPECT_EQ( result.time, result.time_median );
   EXPECT_TRUE( std::isnan( result.time_stddev ) );
}

// minTime makes the loop run longer than the number of loops, which used to write past the end of the samples
TEST( BenchmarksTest, minTimeExceedsLoops )
{
   auto compute = []()
   {
      std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );
   };
   const BenchmarkResult result = runTimeFunction(
      compute,
      1,     // loops
      0.02,  // minTime (s)
      0,     // warmupLoops, 0 for no warmup
      true );  // forceWarmupLoop
   EXPECT_GT( result.loops, 1 );
   EXPECT_GE( result.time * result.loops, 0.02 );
}

TEST( BenchmarksTest, longWarmupIsDiscardedByDefault )
{
   int calls = 0;
   auto compute = [ & ]()
   {
      calls++;
      std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
   };
   const BenchmarkResult result = runTimeFunction(
      compute,
      1,     // loops
      0.01,  // minTime (s)
      1,     // warmupLoops
      true );  // forceWarmupLoop
   EXPECT_EQ( result.loops, 1 );
   EXPECT_EQ( calls, 2 );
}

TEST( BenchmarksTest, longWarmupIsSample )
{
   int calls = 0;
   auto compute = [ & ]()
   {
      calls++;
      std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
   };
   const BenchmarkResult result = runTimeFunction(
      compute,
      1,     // loops
      0.01,  // minTime (s)
      3,     // warmupLoops, the long warmup iteration ends the warmup
      false );  // forceWarmupLoop
   EXPECT_EQ( result.loops, 1 );
   EXPECT_EQ( calls, 1 );
   EXPECT_GE( result.time, 0.01 );
   EXPECT_EQ( result.time, result.time_median );
   EXPECT_TRUE( std::isnan( result.time_stddev ) );
}

TEST( BenchmarksTest, longWarmupCountsTowardsLoops )
{
   int calls = 0;
   auto compute = [ & ]()
   {
      calls++;
      std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
   };
   const BenchmarkResult result = runTimeFunction(
      compute,
      3,     // loops
      0.01,  // minTime (s)
      1,     // warmupLoops
      false );  // forceWarmupLoop
   EXPECT_EQ( result.loops, 3 );
   EXPECT_EQ( calls, 3 );
}

TEST( BenchmarksTest, shortWarmupIsNotSample )
{
   int calls = 0;
   auto compute = [ & ]()
   {
      calls++;
   };
   const BenchmarkResult result = runTimeFunction(
      compute,
      3,    // loops
      0.0,  // minTime (s), the warmup iteration is never a sample without the minimal time
      1,    // warmupLoops
      false );  // forceWarmupLoop
   EXPECT_EQ( result.loops, 3 );
   EXPECT_EQ( calls, 4 );
}

TEST( BenchmarksTest, statistics )
{
   int calls = 0;
   auto compute = [ & ]()
   {
      calls++;
      std::this_thread::sleep_for( std::chrono::milliseconds( calls ) );
   };
   const BenchmarkResult result = runTimeFunction(
      compute,
      4,    // loops
      0.0,  // minTime (s)
      0,    // warmupLoops, 0 for no warmup
      true );  // forceWarmupLoop
   EXPECT_EQ( result.loops, 4 );
   EXPECT_GT( result.time, 0.0 );
   EXPECT_GT( result.time_median, 0.0 );
   EXPECT_GT( result.time_stddev, 0.0 );
}

#include "main.h"
