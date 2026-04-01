// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>

#include "Logging.h"

namespace TNL::Benchmarks {

struct BenchmarkResult
{
   using HeaderElements = Logging::HeaderElements;
   using RowElements = Logging::RowElements;

   virtual ~BenchmarkResult() = default;

   std::size_t loops = 0;
   double time = std::numeric_limits< double >::quiet_NaN();
   double time_stddev = std::numeric_limits< double >::quiet_NaN();
   double cpu_cycles = std::numeric_limits< double >::quiet_NaN();
   double cpu_cycles_stddev = std::numeric_limits< double >::quiet_NaN();
   double bandwidth = std::numeric_limits< double >::quiet_NaN();
   double speedup = std::numeric_limits< double >::quiet_NaN();
   double cpu_cycles_per_operation = 0;
   std::size_t operations_per_loop = 0;

   [[nodiscard]] virtual HeaderElements
   getTableHeader() const
   {
      return HeaderElements(
         { "time",
           "speedup",
           "bandwidth",
           "cycles/op",
           "cycles",
           "time_stddev",
           "time_stddev/time",
           "cycles_stddev",
           "cycles_stddev/cycles",
           "loops",
           "ops_per_loop" } );
   }

   [[nodiscard]] virtual RowElements
   getRowElements() const
   {
      RowElements elements;
      // write in scientific format to avoid precision loss
      elements << std::scientific;

      elements << time;
      if( speedup != 0 )
         elements << speedup;
      else
         elements << "N/A";
      elements << bandwidth;
      if( cpu_cycles_per_operation != 0 )
         elements << cpu_cycles_per_operation;
      else
         elements << "N/A";
      if( cpu_cycles != 0 )
         elements << cpu_cycles;
      else
         elements << "N/A";
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
