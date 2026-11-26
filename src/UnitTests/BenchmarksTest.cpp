#include <chrono>
#include <cmath>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Benchmarks/BenchmarkResult.h>
#include <TNL/Benchmarks/Utils.h>
#include <TNL/Config/parseCommandLine.h>

using namespace TNL;
using namespace TNL::Benchmarks;

// Runs timeFunction on the host.
template< typename ComputeFunction >
BenchmarkResult
runTimeFunction(
   ComputeFunction compute,
   std::size_t loops,
   double maxTime,
   std::size_t warmupLoops,
   double warmupMaxTime,
   bool reuseLongWarmup )
{
   Solvers::IterativeSolverMonitor< double > monitor;
   BenchmarkResult result;
   auto reset = []() {};
   timeFunction< Devices::Host >(
      compute, reset, loops, maxTime, warmupLoops, warmupMaxTime, reuseLongWarmup, monitor, result );
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
      0.0,  // maxTime (s), 0 for no limit
      1,    // warmupLoops
      0.0,  // warmupMaxTime (s), 0 for no limit
      false );  // reuseLongWarmup
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
   // no iterations and no warmup requested, but at least one sample must be measured
   const BenchmarkResult result = runTimeFunction(
      compute,
      0,    // loops
      0.0,  // maxTime (s), 0 for no limit
      0,    // warmupLoops, 0 for no warmup
      0.0,  // warmupMaxTime (s), 0 for no limit
      false );  // reuseLongWarmup
   EXPECT_EQ( result.loops, 1 );
   EXPECT_EQ( calls, 1 );
   EXPECT_GE( result.time, 0.0 );
   EXPECT_EQ( result.time, result.time_median );
   EXPECT_TRUE( std::isnan( result.time_stddev ) );
}

TEST( BenchmarksTest, maxTimeLimitsLoops )
{
   auto compute = []()
   {
      std::this_thread::sleep_for( std::chrono::milliseconds( 5 ) );
   };
   // every iteration takes at least 5 ms, so the maximal time 20 ms is reached after at most 4 iterations
   const BenchmarkResult result = runTimeFunction(
      compute,
      1000,  // loops
      0.02,  // maxTime (s)
      0,     // warmupLoops, 0 for no warmup
      0.0,   // warmupMaxTime (s), 0 for no limit
      false );  // reuseLongWarmup
   EXPECT_GE( result.loops, 1 );
   EXPECT_LE( result.loops, 4 );
}

TEST( BenchmarksTest, warmupMaxTimeLimitsWarmupLoops )
{
   int calls = 0;
   auto compute = [ & ]()
   {
      calls++;
      std::this_thread::sleep_for( std::chrono::milliseconds( 5 ) );
   };
   // every iteration takes at least 5 ms, so the maximal warmup time 20 ms is reached after at most 4 iterations
   const BenchmarkResult result = runTimeFunction(
      compute,
      1,     // loops
      0.0,   // maxTime (s), 0 for no limit
      1000,  // warmupLoops
      0.02,  // warmupMaxTime (s)
      false );  // reuseLongWarmup
   EXPECT_EQ( result.loops, 1 );
   EXPECT_GE( calls, 2 );
   EXPECT_LE( calls, 5 );
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
      10,       // loops
      0.01,     // maxTime (s)
      1,        // warmupLoops
      0.0,      // warmupMaxTime (s), 0 for no limit
      false );  // reuseLongWarmup
   // the first measured iteration exhausts the maximal time
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
      10,    // loops
      0.01,  // maxTime (s)
      3,     // warmupLoops, the long warmup iteration ends the warmup
      0.0,   // warmupMaxTime (s), 0 for no limit
      true );  // reuseLongWarmup
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
   const BenchmarkResult result = runTimeFunction(
      compute,
      3,     // loops
      10.0,  // maxTime (s)
      1,     // warmupLoops
      0.0,   // warmupMaxTime (s), 0 for no limit
      true );  // reuseLongWarmup
   EXPECT_EQ( result.loops, 3 );
   EXPECT_EQ( calls, 4 );
}

TEST( BenchmarksTest, warmupIsNotSampleWithoutMaxTime )
{
   int calls = 0;
   auto compute = [ & ]()
   {
      calls++;
      std::this_thread::sleep_for( std::chrono::milliseconds( 1 ) );
   };
   const BenchmarkResult result = runTimeFunction(
      compute,
      2,    // loops
      0.0,  // maxTime (s), 0 for no limit
      1,    // warmupLoops
      0.0,  // warmupMaxTime (s), 0 for no limit
      true );  // reuseLongWarmup
   EXPECT_EQ( result.loops, 2 );
   EXPECT_EQ( calls, 3 );
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
      0.0,  // maxTime (s), 0 for no limit
      0,    // warmupLoops, 0 for no warmup
      0.0,  // warmupMaxTime (s), 0 for no limit
      false );  // reuseLongWarmup
   EXPECT_EQ( result.loops, 4 );
   EXPECT_GT( result.time, 0.0 );
   EXPECT_GT( result.time_median, 0.0 );
   EXPECT_GT( result.time_stddev, 0.0 );
}

TEST( BenchmarksTest, negativeLoopsAreRejected )
{
   Config::ConfigDescription config;
   Benchmark::configSetup( config );
   for( std::string option : { "--loops", "--warmup-loops" } ) {
      for( std::string value : { "-1", " -1" } ) {
         std::string program = "BenchmarksTest";
         char* argv[] = { program.data(), option.data(), value.data() };
         Config::ParameterContainer parameters;
         EXPECT_FALSE( Config::parseCommandLine( 3, argv, config, parameters, false ) ) << option << " '" << value << "'";
      }
   }
}

TEST( BenchmarksTest, loopsAreParsed )
{
   Config::ConfigDescription config;
   Benchmark::configSetup( config );
   std::string program = "BenchmarksTest";
   std::string loopsOption = "--loops";
   std::string loopsValue = "3";
   std::string warmupOption = "--warmup-loops";
   std::string warmupValue = "0";
   char* argv[] = { program.data(), loopsOption.data(), loopsValue.data(), warmupOption.data(), warmupValue.data() };
   Config::ParameterContainer parameters;
   ASSERT_TRUE( Config::parseCommandLine( 5, argv, config, parameters, false ) );
   EXPECT_EQ( parameters.getParameter< std::size_t >( "loops" ), 3 );
   EXPECT_EQ( parameters.getParameter< std::size_t >( "warmup-loops" ), 0 );
}

#include "main.h"
