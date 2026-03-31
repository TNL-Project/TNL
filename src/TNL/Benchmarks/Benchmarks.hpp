// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <exception>

#include "Benchmarks.h"
#include "JsonLogging.h"
#include "TerminalLogging.h"
#include "Utils.h"

namespace TNL::Benchmarks {

Benchmark::Benchmark( std::ostream& output, std::size_t loops, int verbose )
: loops( loops )
{
   loggers.push_back( std::make_unique< JsonLogging >( output, verbose ) );
   loggers.push_back( std::make_unique< TerminalLogger >( std::cout, verbose ) );
}

void
Benchmark::configSetup( Config::ConfigDescription& config )
{
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< bool >( "reset", "Call reset function between loops.", true );
   config.addEntry< double >( "min-time", "Minimal real time in seconds for every computation.", 0.0 );
   config.addEntry< int >( "verbose", "Verbose mode, the higher number the more verbosity.", 1 );
}

void
Benchmark::setup( const Config::ParameterContainer& parameters )
{
   this->loops = parameters.getParameter< int >( "loops" );
   this->reset = parameters.getParameter< bool >( "reset" );
   this->minTime = parameters.getParameter< double >( "min-time" );
   const int verbose = parameters.getParameter< int >( "verbose" );
   for( auto& logger : loggers )
      logger->setVerbose( verbose );
}

void
Benchmark::setLoops( std::size_t loops )
{
   this->loops = loops;
}

void
Benchmark::setMinTime( double minTime )
{
   this->minTime = minTime;
}

bool
Benchmark::isResettingOn() const
{
   return reset;
}

void
Benchmark::setMetadataColumns( const MetadataColumns& metadata )
{
   for( auto& logger : loggers )
      logger->setMetadataColumns( metadata );
}

void
Benchmark::setMetadataElement( const typename MetadataColumns::value_type& element )
{
   for( auto& logger : loggers )
      logger->setMetadataElement( element );
}

void
Benchmark::setDatasetSize( double datasetSize, double baseTime )
{
   this->datasetSize = datasetSize;
   this->baseTime = baseTime;
}

void
Benchmark::setOperationsPerLoop( std::size_t operationsPerLoop )
{
   this->operations_per_loop = operationsPerLoop;
}

void
Benchmark::setOperation( const std::string& operation, double datasetSize, double baseTime )
{
   monitor.setStage( operation );
   for( auto& logger : loggers )
      logger->setMetadataElement( { "operation", operation }, 0 );
   setDatasetSize( datasetSize, baseTime );
}

template< typename Device, typename ResetFunction, typename ComputeFunction >
void
Benchmark::time( ResetFunction reset, const std::string& performer, ComputeFunction& compute, BenchmarkResult& result )
{
   result.time = std::numeric_limits< double >::quiet_NaN();
   result.time_stddev = std::numeric_limits< double >::quiet_NaN();
   result.cpu_cycles = std::numeric_limits< double >::quiet_NaN();
   result.cpu_cycles_stddev = std::numeric_limits< double >::quiet_NaN();

   // run the monitor main loop
   Solvers::SolverMonitorThread monitor_thread( monitor );
   if( ! loggers.empty() && loggers.front()->getVerbose() <= 1 )
      // stop the main loop when not verbose
      monitor.stopMainLoop();

   std::string errorMessage;
   try {
      if( this->reset ) {
         std::tie( result.loops, result.time, result.time_stddev, result.cpu_cycles, result.cpu_cycles_stddev ) =
            timeFunction< Device >( compute, reset, loops, minTime, monitor );
      }
      else {
         auto noReset = []() {};
         std::tie( result.loops, result.time, result.time_stddev, result.cpu_cycles, result.cpu_cycles_stddev ) =
            timeFunction< Device >( compute, noReset, loops, minTime, monitor );
      }
   }
   catch( const std::exception& e ) {
      errorMessage = "timeFunction failed due to a C++ exception with description: " + std::string( e.what() );
      std::cerr << errorMessage << '\n';
   }

   result.bandwidth = datasetSize / result.time;
   result.speedup = this->baseTime / result.time;
   result.operations_per_loop = this->operations_per_loop;
   if( result.cpu_cycles && this->operations_per_loop )
      result.cpu_cycles_per_operation = result.cpu_cycles / this->operations_per_loop;

   if( this->baseTime == 0.0 )
      this->baseTime = result.time;

   for( auto& logger : loggers )
      logger->logResult( performer, result.getTableHeader(), result.getRowElements(), errorMessage );
}

template< typename Device, typename ResetFunction, typename ComputeFunction >
BenchmarkResult
Benchmark::time( ResetFunction reset, const std::string& performer, ComputeFunction& compute )
{
   BenchmarkResult result;
   time< Device >( reset, performer, compute, result );
   return result;
}

template< typename Device, typename ComputeFunction >
void
Benchmark::time( const std::string& performer, ComputeFunction& compute, BenchmarkResult& result )
{
   auto noReset = []() {};
   time< Device >( noReset, performer, compute, result );
}

template< typename Device, typename ComputeFunction >
BenchmarkResult
Benchmark::time( const std::string& performer, ComputeFunction& compute )
{
   BenchmarkResult result;
   time< Device >( performer, compute, result );
   return result;
}

void
Benchmark::addLogger( std::unique_ptr< Logging > logger )
{
   loggers.push_back( std::move( logger ) );
}

void
Benchmark::addErrorMessage( const std::string& message )
{
   for( auto& logger : loggers )
      logger->writeErrorMessage( message );
   std::cerr << message << '\n';
}

auto
Benchmark::getMonitor() -> SolverMonitorType&
{
   return monitor;
}

double
Benchmark::getBaseTime() const
{
   return baseTime;
}

}  // namespace TNL::Benchmarks
