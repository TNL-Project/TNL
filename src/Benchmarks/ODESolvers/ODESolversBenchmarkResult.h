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

template< typename Solver, typename Device, typename Index >
struct ODESolversBenchmarkResult : public BenchmarkResult
{
   using SolverType = Solver;
   using RealType = typename SolverType::RealType;
   using ValueType = typename SolverType::ValueType;
   using DeviceType = Device;
   using IndexType = Index;
   using ElementType = typename SolverElementType< SolverType >::type;
   using BenchmarkVector = Containers::Vector< ElementType, DeviceType, IndexType >;

   ODESolversBenchmarkResult( const RealType& exactSolution, const SolverType& solver, const BenchmarkVector& benchmarkResult )
   : exactSolution( exactSolution ),
     solver( solver ),
     benchmarkResult( benchmarkResult )
   {}

   HeaderElements
   getTableHeader() const override
   {
      HeaderElements headers = BenchmarkResult::getTableHeader();
      headers.emplace_back( "error" );
      headers.emplace_back( "EOC" );
      headers.emplace_back( "iters" );
      return headers;
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

      RowElements elements = BenchmarkResult::getRowElements();
      // write in scientific format to avoid precision loss
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
