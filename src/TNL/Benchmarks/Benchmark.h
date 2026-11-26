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

/**
 * \brief Base class for running benchmarks with timing and logging support.
 *
 * The Benchmark class provides a unified interface for measuring performance
 * of computational kernels across different devices (CPU, GPU). It supports:
 *
 * - Multiple iterations with automatic loop count determination
 * - Maximum runtime specification, so that long computations are not repeated
 * - Automatic warmup iteration before timing begins
 * - Long computations used as their own warmup, so that they are not executed twice
 * - Configurable output logging
 * - Metadata tracking (device, operation, performer, etc.)
 * - Bandwidth calculations
 * - CPU cycle counting (host devices only)
 *
 * \par Number of iterations
 * The computation is repeated until either the given number of iterations
 * was measured (\ref setLoops) or the measured time reached the maximum runtime
 * (\ref setMaxTime).
 * The maximum runtime does not interrupt a running computation,
 * it only prevents starting the next iteration.
 * Hence, short computations are measured the given number of times
 * and long computations are measured only once.
 * The measurement is preceded by warmup iterations, which are normally discarded
 * and which are limited in the same way (\ref setWarmupLoops, \ref setWarmupMaxTime).
 * When \ref setReuseLongWarmup is set to `true` and the maximum runtime is positive,
 * a warmup iteration taking at least the maximum runtime is used as the only sample instead,
 * so that long computations are not executed once more only to be discarded.
 *
 * Example usage:
 * \code
 * // Configure benchmark
 * TNL::Benchmarks::Benchmark benchmark;
 * TNL::Config::ConfigDescription config;
 * Benchmark::configSetup( config );
 *
 * // Parse command line parameters
 * auto parameters = config.parseCommandLine();
 * benchmark.setup( parameters, argv[ 0 ] );
 *
 * // Set up operation to benchmark
 * benchmark.setMetadataColumns(
 *    Benchmark::MetadataColumns(
 *       {
 *          { "precision", getType< Real >() },
 *          { "size", std::to_string( size ) },
 *       } ) );
 * std::size_t datasetSize = size * sizeof( Real );
 * benchmark.setOperation( "operation-name" );
 * benchmark.setDatasetSize( datasetSize );
 *
 * // Define reset and compute functions
 * auto reset = []() { ... };
 * auto compute = []() { ... };
 * // Run benchmark
 * benchmark.time< Device >( reset, "performer-name", compute );
 * \endcode
 */
class Benchmark
{
public:
   using MetadataElement = Logging::MetadataElement;
   using MetadataColumns = Logging::MetadataColumns;
   using SolverMonitorType = Solvers::IterativeSolverMonitor< double >;

   Benchmark() = default;

   /**
    * \brief Configures benchmark-related command line options.
    *
    * Must be called before parsing command line arguments. Adds the following
    * configuration entries:
    *
    * - `log-file`: Path to JSONL output file (default: `<program>.log`)
    * - `output-mode`: "append" or "overwrite"
    * - `loops`: Maximum number of iterations (default: 10)
    * - `max-time`: Runtime in seconds after which no further iteration is started, 0 for no limit (default: 0.0)
    * - `warmup-loops`: Maximum number of warmup iterations (default: 1)
    * - `warmup-max-time`: Warmup runtime in seconds after which no further warmup iteration is started,
    *   0 for no limit (default: 0.0)
    * - `reuse-long-warmup`: Use a warmup iteration taking at least `max-time` as the only sample (default: false)
    * - `verbose`: Verbosity level (default: 1)
    * - `catch-exceptions`: Catch exceptions during timing (default: true)
    *
    * \param config Reference to configuration description object
    */
   static void
   configSetup( Config::ConfigDescription& config );

   /**
    * \brief Initializes the benchmark from parsed parameters.
    *
    * Extracts benchmark settings from the parameter container and initializes
    * loggers (JSON and/or terminal; only on rank 0 in MPI configurations).
    *
    * If the `log-file` parameter is empty, the log file name defaults to
    * `<programName>.log`, where `programName` is typically derived from
    * `argv[0]`.
    *
    * \param parameters Parsed configuration parameters
    * \param programName Program name used as default log file base name
    */
   void
   setup( const Config::ParameterContainer& parameters, const std::string& programName = "" );

   /**
    * \brief Sets the maximum number of iterations for each measurement.
    *
    * Fewer iterations are executed if the maximum runtime is reached,
    * see \ref setMaxTime. At least one iteration is always measured.
    *
    * \param loops Maximum number of loops to execute
    */
   void
   setLoops( std::size_t loops );

   /**
    * \brief Sets the maximum runtime for measurements.
    *
    * If positive, the benchmark stops iterating as soon as the measured time
    * reaches this value, even if fewer iterations than the loop count were executed.
    * The limit does not interrupt a running computation: it is checked after each iteration
    * and only prevents starting the next one.
    * Hence, the measured time may exceed the limit by up to one iteration
    * and an iteration longer than the limit always runs to its end.
    *
    * \param maxTime Maximum time in seconds (0 for no limit)
    */
   void
   setMaxTime( double maxTime );

   /**
    * \brief Sets the maximum number of warmup iterations.
    *
    * Warmup iterations are executed before timing begins to stabilize
    * thermal and frequency states and amortize one-time costs such as
    * CUDA JIT compilation.
    *
    * \param warmupLoops Maximum number of warmup loops (0 to disable)
    */
   void
   setWarmupLoops( std::size_t warmupLoops );

   /**
    * \brief Sets the maximum warmup runtime.
    *
    * If positive, the warmup stops as soon as this much real time has elapsed,
    * even if fewer iterations than the warmup loop count were executed.
    * Like \ref setMaxTime, the limit does not interrupt a running computation,
    * it only prevents starting the next warmup iteration.
    *
    * \param warmupMaxTime Maximum warmup time in seconds (0 for no limit)
    */
   void
   setWarmupMaxTime( double warmupMaxTime );

   /**
    * \brief Sets whether a long warmup iteration is used as the timing sample.
    *
    * If set to `true` and the maximum runtime (\ref setMaxTime) is positive,
    * a warmup iteration taking at least the maximum runtime ends the warmup
    * and is used as the only timing sample.
    * Such a computation exhausts the time budget by itself and it is long enough
    * that the effects compensated by the warmup are negligible,
    * so it is not executed once more only to be discarded.
    *
    * \param reuseLongWarmup Whether a long warmup iteration is used as the sample (default: `false`)
    */
   void
   setReuseLongWarmup( bool reuseLongWarmup );

   /**
    * \brief Sets metadata columns for all subsequent result rows.
    *
    * Metadata values persist until explicitly changed. Common metadata includes
    * device type, problem size, algorithm variant, etc.
    *
    * \param metadata Vector of key-value pairs to set as metadata
    */
   void
   setMetadataColumns( const MetadataColumns& metadata );

   /**
    * \brief Updates or adds a single metadata element.
    *
    * Useful for incrementally building metadata when running multiple related
    * benchmarks.
    *
    * \param element Key-value pair to set
    */
   void
   setMetadataElement( const typename MetadataColumns::value_type& element );

   /**
    * \brief Sets dataset size for derived metrics.
    *
    * \param datasetSize Dataset size in bytes
    */
   void
   setDatasetSize( std::size_t datasetSize );

   /**
    * \brief Sets the number of operations performed per loop iteration.
    *
    * Used to calculate cycles per operation metric.
    *
    * \param operationsPerLoop Number of operations per loop
    */
   void
   setOperationsPerLoop( std::size_t operationsPerLoop );
   /**
    * \brief Sets the current operation name and dataset size.
    *
    * Operations create vertical divisions in result tables.
    *
    * \param operation Name of the current operation
    */
   void
   setOperation( const std::string& operation );
   /**
    * \brief Times a compute function with reset between iterations.
    *
    * Executes the compute function multiple times, calling `reset()` before each
    * iteration. Results are logged through configured loggers.
    *
    * Warmup iterations (reset + compute) are performed automatically
    * before the timed loop begins, to stabilize thermal and frequency states
    * and amortize one-time costs such as CUDA JIT compilation
    * (see \ref setWarmupLoops).
    * Their times are discarded, unless a long warmup iteration is used as the sample
    * (see \ref setReuseLongWarmup).
    *
    * \tparam Device Device type (e.g., \ref TNL::Devices::Host, \ref TNL::Devices::Cuda)
    * \tparam ResetFunction Callable that resets state before each iteration
    * \tparam ComputeFunction Callable containing the code to benchmark
    *
    * \param reset Function called before each compute iteration
    * \param performer Name identifying the implementation being tested
    * \param compute Function to benchmark
    * \param result Output structure for benchmark results
    */
   template< typename Device, typename ResetFunction, typename ComputeFunction >
   void
   time( ResetFunction reset, const std::string& performer, ComputeFunction& compute, BenchmarkResult& result );

   /**
    * \brief Times a compute function with reset between iterations (returns result).
    *
    * Convenience overload that creates and returns a \ref BenchmarkResult object.
    *
    * \tparam Device Device type
    * \tparam ResetFunction Reset function type
    * \tparam ComputeFunction Compute function type
    *
    * \param reset Function called before each compute iteration
    * \param performer Name identifying the implementation
    * \param compute Function to benchmark
    * \return BenchmarkResult containing timing data
    */
   template< typename Device, typename ResetFunction, typename ComputeFunction >
   BenchmarkResult
   time( ResetFunction reset, const std::string& performer, ComputeFunction& compute );

   /**
    * \brief Times a compute function without explicit reset.
    *
    * Equivalent to calling `time` with an empty reset function.
    *
    * \tparam Device Device type
    * \tparam ComputeFunction Compute function type
    *
    * \param performer Name identifying the implementation
    * \param compute Function to benchmark
    * \param result Output structure for benchmark results
    */
   template< typename Device, typename ComputeFunction >
   void
   time( const std::string& performer, ComputeFunction& compute, BenchmarkResult& result );

   /**
    * \brief Times a compute function without explicit reset (returns result).
    *
    * Equivalent to calling `time` with an empty reset function.
    *
    * \tparam Device Device type
    * \tparam ComputeFunction Compute function type
    *
    * \param performer Name identifying the implementation
    * \param compute Function to benchmark
    * \return BenchmarkResult containing timing data
    */
   template< typename Device, typename ComputeFunction >
   BenchmarkResult
   time( const std::string& performer, ComputeFunction& compute );

   /**
    * \brief Logs an error message through all configured loggers.
    *
    * Should be called when the \ref time method cannot be executed due to
    * errors like memory allocation failures.
    *
    * \param message Error description
    */
   void
   addErrorMessage( const std::string& message );

   /**
    * \brief Adds a logger for outputting benchmark results.
    *
    * Multiple loggers can be added (e.g., both \ref JsonLogging and \ref TerminalLogger).
    *
    * \param logger Unique pointer to a logger object
    */
   void
   addLogger( std::unique_ptr< Logging > logger );

   /**
    * \brief Returns reference to the solver monitor.
    *
    * The monitor tracks iterative solver convergence during benchmarking.
    *
    * \return Reference to SolverMonitorType instance
    */
   [[nodiscard]] SolverMonitorType&
   getMonitor();

   /**
    * \brief Sets whether to catch exceptions during timing of computations.
    *
    * When enabled (default), exceptions thrown during benchmark execution are
    * caught and logged as error messages. When disabled, exceptions propagate
    * up to the caller, allowing for stricter error handling in automated tests.
    *
    * \param catchExceptions \c true to catch exceptions (default), \c false to propagate them
    */
   void
   setCatchExceptions( bool catchExceptions );

   /**
    * \brief Returns whether exceptions are caught during timing.
    *
    * \return \c true if exceptions are caught, \c false if they propagate
    */
   [[nodiscard]] bool
   getCatchExceptions() const;

protected:
   using BenchmarkLoggers = std::vector< std::unique_ptr< Logging > >;

   BenchmarkLoggers loggers;
   std::ofstream logFile;

   std::size_t loops = 10;

   std::size_t warmupLoops = 1;

   double warmupMaxTime = 0.0;

   bool reuseLongWarmup = false;

   std::size_t operations_per_loop = 0;

   double maxTime = 0.0;

   std::size_t datasetSize = 0;

   bool catchExceptions = true;

   SolverMonitorType monitor;
};

}  // namespace TNL::Benchmarks

#include "Benchmark.hpp"
