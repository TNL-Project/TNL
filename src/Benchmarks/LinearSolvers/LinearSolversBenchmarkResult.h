// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Solvers/Linear/Utils/LinearResidueGetter.h>

// BenchmarkResult to add extra columns to the benchmark
// (solved status, iterations, preconditioned residue, true residue)
template< typename Vector, typename Matrix, template< typename > class Solver >
struct LinearSolversBenchmarkResult : public TNL::Benchmarks::BenchmarkResult
{
   using SolverType = Solver< Matrix >;

   SolverType& solver;
   const std::shared_ptr< Matrix >& matrix;
   const Vector& x;
   const Vector& b;

   LinearSolversBenchmarkResult( SolverType& solver, const std::shared_ptr< Matrix >& matrix, const Vector& x, const Vector& b )
   : solver( solver ),
     matrix( matrix ),
     x( x ),
     b( b )
   {}

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      HeaderElements elements = TNL::Benchmarks::BenchmarkResult::getTableHeader();
      elements.emplace_back( "solved" );
      elements.emplace_back( "iterations" );
      elements.emplace_back( "residue precond" );
      elements.emplace_back( "residue true" );
      return elements;
   }

   [[nodiscard]] std::vector< int >
   getColumnWidthHints() const override
   {
      auto hints = TNL::Benchmarks::BenchmarkResult::getColumnWidthHints();
      hints.emplace_back( 8 );   // solved
      hints.emplace_back( 14 );  // iterations
      hints.emplace_back( 14 );  // residue precond
      hints.emplace_back( 14 );  // residue true
      return hints;
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      RowElements elements = TNL::Benchmarks::BenchmarkResult::getRowElements();

      const bool converged = solver.checkConvergence();
      const long iterations = solver.getIterations();
      const double residue_precond = solver.getResidue();
      const double residue_true = TNL::Solvers::Linear::LinearResidueGetter::getResidue( *this->matrix, x, b );

      elements << ( converged ? "yes" : "no" ) << iterations << residue_precond << residue_true;
      return elements;
   }
};
