// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Benchmarks/Benchmarks.h>

namespace TNL::Benchmarks {

template< typename Solver, bool isStatic = Solver::isStatic() >
struct SolverElementType
{
   using type = typename Solver::ValueType;
};

template< typename Solver >
struct SolverElementType< Solver, true >
{
   using type = typename Solver::VectorType;
};

template< typename Solver, typename Device, typename Index, typename Logger = JsonLogging >
struct ODESolversBenchmarkResult : public BenchmarkResult
{
   using SolverType = Solver;
   using RealType = typename SolverType::RealType;
   using ValueType = typename SolverType::ValueType;
   using DeviceType = Device;
   using IndexType = Index;
   using ElementType = typename SolverElementType< SolverType >::type;
   using BenchmarkVector = Containers::Vector< ElementType, DeviceType, IndexType >;

   using BenchmarkResult::bandwidth;
   using BenchmarkResult::speedup;
   using BenchmarkResult::time;
   using BenchmarkResult::time_stddev;
   using typename BenchmarkResult::HeaderElements;
   using typename BenchmarkResult::RowElements;

   ODESolversBenchmarkResult( const RealType& exactSolution, const SolverType& solver, const BenchmarkVector& benchmarkResult )
   : exactSolution( exactSolution ),
     solver( solver ),
     benchmarkResult( benchmarkResult )
   {}

   HeaderElements
   getTableHeader() const override
   {
      return HeaderElements(
         { "time", "time_stddev", "time_stddev/time", "loops", "bandwidth", "speedup", "error", "EOC", "iters" } );
   }

   std::vector< int >
   getColumnWidthHints() const override
   {
      return std::vector< int >( { 14, 14, 14, 6, 14, 10, 14, 10, 14 } );
   }

   RowElements
   getRowElements() const override
   {
      RealType error;
      if constexpr( SolverType::isStatic() )
         error = abs( exactSolution - benchmarkResult.getElement( 0 )[ 0 ] );
      else
         error = abs( exactSolution - benchmarkResult.getElement( 0 ) );
      RealType eoc = -1.0;
      if( lastError != -1.0 )
         eoc = log( lastError / error ) / log( 2.0 );
      lastError = error;

      RowElements elements;
      // write in scientific format to avoid precision loss
      elements << std::scientific << time << time_stddev << time_stddev / time << loops << bandwidth;
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

   void
   reset()
   {
      this->lastError = -1.0;
   }

protected:
   RealType exactSolution;
   mutable RealType lastError = -1.0;
   const SolverType& solver;
   const BenchmarkVector& benchmarkResult;
};

}  // namespace TNL::Benchmarks
