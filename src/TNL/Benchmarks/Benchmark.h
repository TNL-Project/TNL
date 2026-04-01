// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <memory>
#include <vector>
#include <fstream>

#include "BenchmarkResult.h"
#include "Logging.h"

#include <TNL/String.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>

namespace TNL::Benchmarks {

inline constexpr double oneGB = 1024.0 * 1024.0 * 1024.0;

class Benchmark
{
public:
   using MetadataElement = Logging::MetadataElement;
   using MetadataColumns = Logging::MetadataColumns;
   using SolverMonitorType = Solvers::IterativeSolverMonitor< double >;

   Benchmark() = default;

   static void
   configSetup( Config::ConfigDescription& config );

   void
   setup( const Config::ParameterContainer& parameters );

   void
   setLoops( std::size_t loops );

   void
   setMinTime( double minTime );

   // Sets metadata columns -- values used for all subsequent rows until
   // the next call to this function.
   void
   setMetadataColumns( const MetadataColumns& metadata );

   // Sets the value of one metadata column -- useful for iteratively
   // changing MetadataColumns that were set using the previous method.
   void
   setMetadataElement( const typename MetadataColumns::value_type& element );

   // Sets the dataset size and base time for the calculations of bandwidth
   // and speedup in the benchmarks result.
   void
   setDatasetSize(
      double datasetSize = 0.0,  // in GB
      double baseTime = 0.0 );

   void
   setOperationsPerLoop( std::size_t operationsPerLoop );

   // Sets current operation -- operations expand the table vertically
   //  - baseTime should be reset to 0.0 for most operations, but sometimes
   //    it is useful to override it
   //  - Order of operations inside a "Benchmark" does not matter, rows can be
   //    easily sorted while converting to HTML.)
   void
   setOperation(
      const std::string& operation,
      double datasetSize = 0.0,  // in GB
      double baseTime = 0.0 );

   // Times a single ComputeFunction. Subsequent calls implicitly split
   // the current operation into sub-columns identified by "performer",
   // which are further split into "bandwidth", "time" and "speedup" columns.
   template< typename Device, typename ResetFunction, typename ComputeFunction >
   void
   time( ResetFunction reset, const std::string& performer, ComputeFunction& compute, BenchmarkResult& result );

   template< typename Device, typename ResetFunction, typename ComputeFunction >
   BenchmarkResult
   time( ResetFunction reset, const std::string& performer, ComputeFunction& compute );

   // The same methods as above but without the reset function
   template< typename Device, typename ComputeFunction >
   void
   time( const std::string& performer, ComputeFunction& compute, BenchmarkResult& result );

   template< typename Device, typename ComputeFunction >
   BenchmarkResult
   time( const std::string& performer, ComputeFunction& compute );

   // Adds an error message to the log. Should be called in places where the
   // "time" method could not be called (e.g. due to failed allocation).
   void
   addErrorMessage( const std::string& message );

   void
   addLogger( std::unique_ptr< Logging > logger );

   [[nodiscard]] SolverMonitorType&
   getMonitor();

   [[nodiscard]] double
   getBaseTime() const;

protected:
   using BenchmarkLoggers = std::vector< std::unique_ptr< Logging > >;

   BenchmarkLoggers loggers;
   std::ofstream logFile;

   std::size_t loops = 10;

   std::size_t operations_per_loop = 0;

   double minTime = 0.0;

   double datasetSize = 0.0;

   double baseTime = 0.0;

   SolverMonitorType monitor;
};

}  // namespace TNL::Benchmarks

#include "Benchmark.hpp"
