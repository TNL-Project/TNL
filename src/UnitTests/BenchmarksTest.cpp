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
runTimeFunction(
   ComputeFunction compute,
   std::size_t minLoops,
   std::size_t maxLoops,
   double minTime,
   double maxTime,
   std::size_t warmupLoops )
{
   Solvers::IterativeSolverMonitor< double > monitor;
   BenchmarkResult result;
   auto reset = []() {};
   timeFunction< Devices::Host >( compute, reset, minLoops, maxLoops, minTime, maxTime, warmupLoops, 0.0, monitor, result );
   return result;
}

TEST( BenchmarksTest, minLoops )
{
   int calls = 0;
   auto compute = [ & ]()
   {
      calls++;
   };
   const BenchmarkResult result = runTimeFunction( compute, 5, 0, 0.0, 0.0, 1 );
   EXPECT_EQ( result.loops, 5 );
   // one warmup iteration
   EXPECT_EQ( calls, 6 );
}

TEST( BenchmarksTest, minTimeExceedsMinLoops )
{
   auto compute = []()
   {
      std::this_thread::sleep_for( std::chrono::milliseconds( 2 ) );
   };
   const BenchmarkResult result = runTimeFunction( compute, 2, 0, 0.02, 0.0, 0 );
   EXPECT_GT( result.loops, 2 );
   EXPECT_GE( result.time * result.loops, 0.02 );
}

TEST( BenchmarksTest, maxLoopsLimitsMinTime )
{
   auto compute = []() {};
   const BenchmarkResult result = runTimeFunction( compute, 1, 3, 1000.0, 0.0, 0 );
   EXPECT_EQ( result.loops, 3 );
}

TEST( BenchmarksTest, maxLoopsLimitsMinLoops )
{
   auto compute = []() {};
   const BenchmarkResult result = runTimeFunction( compute, 10, 3, 0.0, 0.0, 0 );
   EXPECT_EQ( result.loops, 3 );
}

TEST( BenchmarksTest, maxTimeLimitsMinLoops )
{
   auto compute = []()
   {
      std::this_thread::sleep_for( std::chrono::milliseconds( 5 ) );
   };
   // every iteration takes at least 5 ms, so the maximal time 20 ms is reached after at most 4 iterations
   const BenchmarkResult result = runTimeFunction( compute, 1000, 0, 0.0, 0.02, 0 );
   EXPECT_GE( result.loops, 1 );
   EXPECT_LE( result.loops, 4 );
}

TEST( BenchmarksTest, longWarmupIsTheOnlySample )
{
   int calls = 0;
   auto compute = [ & ]()
   {
      calls++;
      std::this_thread::sleep_for( std::chrono::milliseconds( 20 ) );
   };
   const BenchmarkResult result = runTimeFunction( compute, 10, 0, 1.0, 0.01, 1 );
   EXPECT_EQ( result.loops, 1 );
   EXPECT_EQ( calls, 1 );
   EXPECT_GE( result.time, 0.01 );
   EXPECT_EQ( result.time, result.time_median );
   EXPECT_TRUE( std::isnan( result.time_stddev ) );
}

TEST( BenchmarksTest, shortWarmupIsNotSample )
{
   int calls = 0;
   auto compute = [ & ]()
   {
      calls++;
   };
   const BenchmarkResult result = runTimeFunction( compute, 3, 0, 0.0, 10.0, 1 );
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
   const BenchmarkResult result = runTimeFunction( compute, 4, 4, 0.0, 0.0, 0 );
   EXPECT_EQ( result.loops, 4 );
   EXPECT_GT( result.time, 0.0 );
   EXPECT_GT( result.time_median, 0.0 );
   EXPECT_GT( result.time_stddev, 0.0 );
}

#include "main.h"
