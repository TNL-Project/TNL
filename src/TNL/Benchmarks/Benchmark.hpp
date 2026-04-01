// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <exception>

#include <TNL/MPI/Comm.h>

#include "Benchmark.h"
#include "JsonLogging.h"
#include "TerminalLogging.h"
#include "Utils.h"

namespace TNL::Benchmarks {

void
Benchmark::configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General benchmark settings:" );
   config.addEntry< String >( "log-file", "Log file name for JSON output.", "" );
   config.addEntry< String >( "output-mode", "Mode for opening the log file.", "overwrite" );
   config.addEntryEnum( "append" );
   config.addEntryEnum( "overwrite" );
   config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
   config.addEntry< bool >( "reset", "Call reset function between loops.", true );
   config.addEntry< double >( "min-time", "Minimal real time in seconds for every computation.", 0.0 );
   config.addEntry< int >( "verbose", "Verbose mode for terminal output, the higher number the more verbosity.", 1 );
}

void
Benchmark::setup( const Config::ParameterContainer& parameters )
{
   this->loops = parameters.getParameter< int >( "loops" );
   this->reset = parameters.getParameter< bool >( "reset" );
   this->minTime = parameters.getParameter< double >( "min-time" );
   const int verbose = parameters.getParameter< int >( "verbose" );

   // Only root rank initializes loggers
   const int rank = TNL::MPI::GetRank();
   if( rank > 0 )
      return;

   // Set up JSON logging if log-file is specified
   const String& logFileName = parameters.getParameter< String >( "log-file" );
   if( ! logFileName.empty() ) {
      const String& outputMode = parameters.getParameter< String >( "output-mode" );
      auto mode = std::ios::out;
      if( outputMode == "append" )
         mode |= std::ios::app;
      logFile.open( logFileName.getString(), mode );
      addLogger( std::make_unique< JsonLogging >( logFile, verbose ) );

      // Write global metadata into a separate file
      std::map< std::string, std::string > metadata = getHardwareMetadata();
      writeMapAsJson( metadata, logFileName, ".metadata.json" );
   }

   // Set up terminal logging
   addLogger( std::make_unique< TerminalLogger >( std::cout, verbose ) );
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
   // run the monitor main loop
   Solvers::SolverMonitorThread monitor_thread( monitor );
   if( ! loggers.empty() && loggers.front()->getVerbose() <= 1 )
      // stop the main loop when not verbose
      monitor.stopMainLoop();

   std::string errorMessage;
   try {
      if( this->reset ) {
         timeFunction< Device >( compute, reset, loops, minTime, monitor, result );
      }
      else {
         auto noReset = []() {};
         timeFunction< Device >( compute, noReset, loops, minTime, monitor, result );
      }
   }
   catch( const std::exception& e ) {
      errorMessage = "timeFunction failed due to a C++ exception with description: " + std::string( e.what() );
      std::cerr << errorMessage << '\n';
   }

   result.setDerivedResults( datasetSize, baseTime, operations_per_loop );

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
