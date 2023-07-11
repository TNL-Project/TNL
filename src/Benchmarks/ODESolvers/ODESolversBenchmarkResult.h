#pragma once

#include <TNL/Benchmarks/Benchmarks.h>

namespace TNL::Benchmarks {

template< typename Real,
          typename Device,
          typename Index,
          typename ResultReal = Real,
          typename Logger = JsonLogging >
struct ODESolversBenchmarkResult
: public BenchmarkResult
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using BenchmarkVector = Containers::Vector< ResultReal, Device, Index >;

   using typename BenchmarkResult::HeaderElements;
   using typename BenchmarkResult::RowElements;
   using BenchmarkResult::stddev;
   using BenchmarkResult::bandwidth;
   using BenchmarkResult::speedup;
   using BenchmarkResult::time;


   ODESolversBenchmarkResult( const RealType& exactSolution,
                              const BenchmarkVector& benchmarkResult )
   : exactSolution( exactSolution ), benchmarkResult( benchmarkResult )
   {}

   virtual HeaderElements getTableHeader() const override {
      return HeaderElements({ "time", "stddev", "stddev/time", "loops", "bandwidth", "speed-up", "error", "EOC" });
   }

   virtual std::vector< int > getColumnWidthHints() const override {
      return std::vector< int >({ 14, 14, 14, 6, 14, 10, 14, 10 });
   }

   virtual RowElements getRowElements() override {
      auto error = abs( exactSolution - benchmarkResult.getElement( 0 ) );
      RealType eoc = -1.0;
      if( lastError != -1.0 )
         eoc = log( lastError / error ) / log( 2.0 );
      lastError = error;

      RowElements elements;
      // write in scientific format to avoid precision loss
      elements << std::scientific << time << stddev << stddev/time << loops << bandwidth;
      elements << std::fixed;
      if( speedup != 0.0 )
         elements << speedup;
      else
         elements << "N/A";
      elements << std::scientific << error;
      elements << std::fixed;
      if( eoc != -1.0 )
         elements << eoc;
      else
         elements << "N/A";
      return elements;
   }

   void reset() { this->lastError = -1.0; }

protected:
   RealType exactSolution, lastError = -1.0;
   const BenchmarkVector& benchmarkResult;
};

} // namespace TNL::Benchmarks
