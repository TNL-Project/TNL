// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>

#include "Logging.h"

namespace TNL::Benchmarks {

/**
 * \brief Container for benchmark measurement results.
 *
 * Stores timing data and derived metrics from benchmark runs. The class
 * supports virtual methods to allow derived classes to customize result
 * storage and formatting.
 *
 * Measured values:
 *
 * - `loops`: Number of iterations executed
 * - `time`: Mean execution time per iteration
 * - `time_median`: Median execution time per iteration
 * - `time_stddev`: Standard deviation of execution time
 * - `cpu_cycles`: Mean CPU cycles per iteration (host devices only)
 * - `cpu_cycles_median`: Median CPU cycles per iteration (host devices only)
 * - `cpu_cycles_stddev`: Standard deviation of CPU cycles
 *
 * Derived values (computed by \ref setDerivedResults):
 *
 * - `bandwidth`: Dataset size divided by execution time (GB/s)
 * - `cpu_cycles_per_operation`: CPU cycles divided by operations per loop
 */
struct BenchmarkResult
{
   using HeaderElements = Logging::HeaderElements;
   using RowElements = Logging::RowElements;

   virtual ~BenchmarkResult() = default;

   std::size_t loops = 0;
   double time = std::numeric_limits< double >::quiet_NaN();
   double time_median = std::numeric_limits< double >::quiet_NaN();
   double time_stddev = std::numeric_limits< double >::quiet_NaN();
   double cpu_cycles = std::numeric_limits< double >::quiet_NaN();
   double cpu_cycles_median = std::numeric_limits< double >::quiet_NaN();
   double cpu_cycles_stddev = std::numeric_limits< double >::quiet_NaN();
   double bandwidth = std::numeric_limits< double >::quiet_NaN();
   double cpu_cycles_per_operation = 0;
   std::size_t operations_per_loop = 0;

   /**
    * \brief Stores raw timing results.
    *
    * Called by `timeFunction` after completing all iterations. Derived classes
    * can override to customize how timing data is stored.
    *
    * \param loops_ Number of iterations executed
    * \param time_ Mean execution time per iteration
    * \param time_median_ Median execution time per iteration
    * \param time_stddev_ Standard deviation of execution time
    * \param cpu_cycles_ Mean CPU cycles per iteration
    * \param cpu_cycles_median_ Median CPU cycles per iteration
    * \param cpu_cycles_stddev_ Standard deviation of CPU cycles
    */
   virtual void
   setTimeResults(
      std::size_t loops_,
      double time_,
      double time_median_,
      double time_stddev_,
      double cpu_cycles_,
      double cpu_cycles_median_,
      double cpu_cycles_stddev_ )
   {
      loops = loops_;
      time = time_;
      time_median = time_median_;
      time_stddev = time_stddev_;
      cpu_cycles = cpu_cycles_;
      cpu_cycles_median = cpu_cycles_median_;
      cpu_cycles_stddev = cpu_cycles_stddev_;
   }

   /**
    * \brief Computes derived metrics from raw timing data.
    *
    * Must be called after \ref setTimeResults. Calculates bandwidth and
    * cycles per operation based on dataset size and operation count.
    *
    * \param datasetSize Dataset size in gigabytes
    * \param operationsPerLoop Number of operations performed per iteration
    */
   virtual void
   setDerivedResults( double datasetSize, std::size_t operationsPerLoop )
   {
      bandwidth = datasetSize / time;
      operations_per_loop = operationsPerLoop;
      if( cpu_cycles != 0.0 && operationsPerLoop != 0 )
         cpu_cycles_per_operation = cpu_cycles / operationsPerLoop;
   }

   /**
    * \brief Returns table header column names.
    *
    * \return Vector of column names in display order
    */
   [[nodiscard]] virtual HeaderElements
   getTableHeader() const
   {
      return HeaderElements(
         { "time",
           "time_median",
           "bandwidth",
           "cycles/op",
           "cycles",
           "cycles_median",
           "time_stddev",
           "time_stddev/time",
           "cycles_stddev",
           "cycles_stddev/cycles",
           "loops",
           "ops_per_loop" } );
   }

   /**
    * \brief Returns formatted row data for logging.
    *
    * Formats all result values as strings using scientific notation for
    * floating-point values. Missing or invalid values are shown as "N/A".
    *
    * \return \ref LoggingRowElements containing formatted string values
    */
   [[nodiscard]] virtual RowElements
   getRowElements() const
   {
      RowElements elements;
      // write in scientific format to avoid precision loss
      elements << std::scientific;

      elements << time << time_median;
      elements << bandwidth;
      if( cpu_cycles_per_operation != 0 )
         elements << cpu_cycles_per_operation;
      else
         elements << "N/A";
      if( cpu_cycles != 0 )
         elements << cpu_cycles << cpu_cycles_median;
      else
         elements << "N/A"
                  << "N/A";
      elements << time_stddev << time_stddev / time;
      if( cpu_cycles != 0 )
         elements << cpu_cycles_stddev << cpu_cycles_stddev / cpu_cycles;
      else
         elements << "N/A"
                  << "N/A";
      elements << loops;
      if( operations_per_loop != 0 )
         elements << operations_per_loop;
      else
         elements << "N/A";
      return elements;
   }
};

}  // namespace TNL::Benchmarks
