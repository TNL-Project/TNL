// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Benchmarks/Benchmarks.h>

// BenchmarkResult to add extra columns to the benchmark
// (iterations, preconditioned residue, true residue)
template< typename Vector, typename Matrix, template< typename > class Solver >
struct BenchmarkResult : public TNL::Benchmarks::BenchmarkResult
{
   using HeaderElements = BenchmarkResult::HeaderElements;
   using RowElements = BenchmarkResult::RowElements;
   using SolverType = Solver< Matrix >;

   Solver< Matrix >& solver;
   const std::shared_ptr< Matrix >& matrix;
   const Vector& x;
   const Vector& b;

   BenchmarkResult( Solver< Matrix >& solver, const std::shared_ptr< Matrix >& matrix, const Vector& x, const Vector& b )
   : solver( solver ),
     matrix( matrix ),
     x( x ),
     b( b )
   {}

   [[nodiscard]] virtual HeaderElements
   getTableHeader() const override
   {
      return HeaderElements(
         { "time", "speedup", "stddev", "stddev/time", "solved", "iterations", "residue_precond", "residue_true" } );
   }

   [[nodiscard]] virtual std::vector< int >
   getColumnWidthHints() const override
   {
      return std::vector< int >( { 14,      // time
                                   8,       // speedup
                                   16,      // time_stddev
                                   18,      // time_stddev/time
                                   8,       // solved
                                   14,      // iterations
                                   14,      // residue precond
                                   14 } );  // residue true
   }

   virtual RowElements
   getRowElements() const override
   {
      RowElements elements;
      if constexpr( SolverType::isIterativeSolver() ) {
         const bool converged = ! std::isnan( solver.getResidue() ) && solver.getResidue() < solver.getConvergenceResidue();
         const long iterations = solver.getIterations();
         const double residue_precond = solver.getResidue();

         Vector r;
         r.setLike( x );
         matrix->vectorProduct( x, r );
         r = b - r;
         const double residue_true = lpNorm( r, 2.0 ) / lpNorm( b, 2.0 );

         elements << time;
         if( speedup != 0 )
            elements << speedup;
         else
            elements << "N/A";
         elements << time_stddev << time_stddev / time;
         elements << ( converged ? "yes" : "no" ) << iterations << residue_precond << residue_true;
      }
      else {  // direct solver
         const bool solved = solver.solved();
         RowElements elements;
         elements << time;
         if( speedup != 0 )
            elements << speedup;
         else
            elements << "N/A";
         elements << time_stddev << time_stddev / time;
         elements << ( solved ? "yes" : "no" ) << "N/A" << "N/A" << "N/A";
      }
      return elements;
   }
};
