// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename PrecisionType >
struct EigenBenchmarkResult : TNL::Benchmarks::BenchmarkResult
{
   EigenBenchmarkResult( const PrecisionType& epsilon, const int& iterations, const double& error )
   : epsilon( epsilon ), iterations( iterations ), error( error )
   {}

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      return HeaderElements( { "time", "stddev", "stddev/time", "loops", "epsilon", "iterations", "error" } );
   }

   [[nodiscard]] std::vector< int >
   getColumnWidthHints() const override
   {
      return std::vector< int >( { 14, 14, 14, 6, 14, 12, 14 } );
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      RowElements elements;
      // write in scientific format to avoid precision loss
      elements << std::scientific << time << time_stddev << time_stddev / time << loops << epsilon << ( iterations / loops )
               << ( error / loops );
      return elements;
   }

   const PrecisionType& epsilon;
   const int& iterations;
   const double& error;
};

template< typename Device >
const char*
performer()
{
   if( std::is_same< Device, TNL::Devices::Host >::value )
#ifdef HAVE_OPENMP
      return "CPUP";
#else
      return "CPU";
#endif
   else if( std::is_same< Device, TNL::Devices::Cuda >::value )
      return "GPU";
   else
      return "unknown";
}
