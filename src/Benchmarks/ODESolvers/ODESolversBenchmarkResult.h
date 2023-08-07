#pragma once

#include <TNL/Benchmarks/Benchmarks.h>

namespace TNL::Benchmarks {

template< typename Solver,
          typename ResultReal = typename Solver::RealType,
          typename Logger = JsonLogging >
struct ODESolversBenchmarkResult
: public BenchmarkResult
{
   using SolverType = Solver;
   using RealType = typename SolverType::RealType;
   using DeviceType = typename SolverType::DeviceType;
   using IndexType = typename SolverType::IndexType;
   using BenchmarkVector = Containers::Vector< ResultReal, DeviceType, IndexType >;

   using typename BenchmarkResult::HeaderElements;
   using typename BenchmarkResult::RowElements;
   using BenchmarkResult::stddev;
   using BenchmarkResult::bandwidth;
   using BenchmarkResult::speedup;
   using BenchmarkResult::time;


   ODESolversBenchmarkResult( const RealType& exactSolution,
                              const SolverType& solver,
                              const BenchmarkVector& benchmarkResult )
   : exactSolution( exactSolution ), solver( solver ), benchmarkResult( benchmarkResult )
   {}

   virtual HeaderElements getTableHeader() const override {
      return HeaderElements({ "time", "stddev", "stddev/time", "loops", "bandwidth", "speed-up", "error", "EOC", "iters" });
   }

   virtual std::vector< int > getColumnWidthHints() const override {
      return std::vector< int >({ 14, 14, 14, 6, 14, 10, 14, 10, 14 });
   }

   virtual RowElements getRowElements() const override {
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
      elements << solver.getIterations();
      return elements;
   }

   void reset() { this->lastError = -1.0; }

protected:
   RealType exactSolution;
   mutable RealType lastError = -1.0;
   const SolverType& solver;
   const BenchmarkVector& benchmarkResult;
};

} // namespace TNL::Benchmarks
