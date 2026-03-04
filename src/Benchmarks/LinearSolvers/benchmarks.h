// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>
#include <TNL/Matrices/DistributedMatrix.h>

#include <stdexcept>  // std::runtime_error
#include "BenchmarkResults.h"

template< typename Device >
const char*
getPerformer()
{
   if( std::is_same_v< Device, TNL::Devices::Cuda > )
      return "GPU";
   return "CPU";
}

template< typename Matrix >
void
barrier( const Matrix& matrix )
{}

template< typename Matrix >
void
barrier( const TNL::Matrices::DistributedMatrix< Matrix >& matrix )
{
   TNL::MPI::Barrier( matrix.getCommunicator() );
}

template< typename Device >
bool
checkDevice( const TNL::Config::ParameterContainer& parameters )
{
   const auto device = parameters.getParameter< TNL::String >( "devices" );
   if( device == "all" )
      return true;
   if( std::is_same_v< Device, TNL::Devices::Host > && device == "host" )
      return true;
   if( std::is_same_v< Device, TNL::Devices::Cuda > && device == "cuda" )
      return true;
   return false;
}

template< template< typename > class Solver, typename Matrix >
void
benchmarkSolverSetup( TNL::Benchmarks::Benchmark<>& benchmark,
                      const TNL::Config::ParameterContainer& parameters,
                      const std::shared_ptr< Matrix >& matrix,
                      const std::string& solver_name )
{
   // skip benchmarks on devices which the user did not select
   if( ! checkDevice< typename Matrix::DeviceType >( parameters ) )
      return;

   barrier( matrix );
   const char* performer = getPerformer< typename Matrix::DeviceType >();

   Solver< Matrix > solver;
   solver.setup( parameters );

   auto compute = [ & ]()
   {
      solver.setMatrix( matrix );
      barrier( matrix );
   };

   benchmark.setOperation( solver_name + " setMatrix" );
   benchmark.time< typename Matrix::DeviceType >( performer, compute );
}

template< template< typename > class Preconditioner, typename Matrix >
void
benchmarkPreconditionerUpdate( TNL::Benchmarks::Benchmark<>& benchmark,
                               const TNL::Config::ParameterContainer& parameters,
                               const std::shared_ptr< Matrix >& matrix,
                               const std::string& preconditioner_name )
{
   // skip benchmarks on devices which the user did not select
   if( ! checkDevice< typename Matrix::DeviceType >( parameters ) )
      return;

   barrier( matrix );
   const char* performer = getPerformer< typename Matrix::DeviceType >();
   Preconditioner< Matrix > preconditioner;
   preconditioner.setup( parameters );

   auto compute = [ & ]()
   {
      preconditioner.update( matrix );
      barrier( matrix );
   };

   benchmark.setOperation( preconditioner_name + " preconditioner update" );
   benchmark.time< typename Matrix::DeviceType >( performer, compute );
}

template< typename >
struct NoPreconditioner
{};

template< template< typename > class Solver,
          template< typename > class Preconditioner = NoPreconditioner,
          typename Matrix,
          typename Vector >
void
benchmarkSolver( TNL::Benchmarks::Benchmark<>& benchmark,
                 const TNL::Config::ParameterContainer& parameters,
                 const std::shared_ptr< Matrix >& matrix,
                 const Vector& x0,
                 const Vector& b,
                 const std::string& solver_name )
{
   // skip benchmarks on devices which the user did not select
   if( ! checkDevice< typename Matrix::DeviceType >( parameters ) )
      return;

   barrier( matrix );
   const char* performer = getPerformer< typename Matrix::DeviceType >();

   // setup
   Solver< Matrix > solver;
   solver.setup( parameters );
   solver.setMatrix( matrix );

   solver.setSolverMonitor( benchmark.getMonitor() );

   if constexpr( ! std::is_same_v< Preconditioner< Matrix >, NoPreconditioner< Matrix > > ) {
      auto pre = std::make_shared< Preconditioner< Matrix > >();
      pre->setup( parameters );
      solver.setPreconditioner( pre );
      // preconditioner update may throw if it's not implemented for CUDA
      try {
         pre->update( matrix );
      }
      catch( const TNL::Exceptions::NotImplementedError& ) {
         return;
      }
   }

   Vector x;
   x.setLike( x0 );

   // reset function
   auto reset = [ & ]()
   {
      x = x0;
   };

   // benchmark function
   auto compute = [ & ]()
   {
      const bool solved = solver.solve( b, x );
      barrier( matrix );
      if( ! solved )
         throw std::runtime_error( "solver failed or did not converge" );
   };

   BenchmarkResult< Vector, Matrix, Solver > benchmarkResult( solver, matrix, x, b );
   benchmark.setOperation( solver_name );
   benchmark.time< typename Matrix::DeviceType >( reset, performer, compute, benchmarkResult );
}

template< template< typename > class Solver, typename Matrix, typename Vector >
void
benchmarkDirectSolver( TNL::Benchmarks::Benchmark<>& benchmark,
                       const TNL::Config::ParameterContainer& parameters,
                       const std::shared_ptr< Matrix >& matrix,
                       const Vector& x0,
                       const Vector& b,
                       const std::string& solver_name )
{
   benchmarkSolverSetup< Solver >( benchmark, parameters, matrix, solver_name );
   benchmarkSolver< Solver >( benchmark, parameters, matrix, x0, b, solver_name + " solve" );
}
