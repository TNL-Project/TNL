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

template< template< typename > class Preconditioner, typename Matrix >
void
benchmarkPreconditionerUpdate( TNL::Benchmarks::Benchmark<>& benchmark,
                               const TNL::Config::ParameterContainer& parameters,
                               const std::shared_ptr< Matrix >& matrix )
{
   // skip benchmarks on devices which the user did not select
   if( ! checkDevice< typename Matrix::DeviceType >( parameters ) )
      return;

   barrier( matrix );
   const char* performer = getPerformer< typename Matrix::DeviceType >();
   Preconditioner< Matrix > preconditioner;
   preconditioner.setup( parameters );

   auto reset = []() {};
   auto compute = [ & ]()
   {
      preconditioner.update( matrix );
      barrier( matrix );
   };

   benchmark.time< typename Matrix::DeviceType >( reset, performer, compute );
}

template< template< typename > class Solver, template< typename > class Preconditioner, typename Matrix, typename Vector >
void
benchmarkSolver( TNL::Benchmarks::Benchmark<>& benchmark,
                 const TNL::Config::ParameterContainer& parameters,
                 const std::shared_ptr< Matrix >& matrix,
                 const Vector& x0,
                 const Vector& b )
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

   auto pre = std::make_shared< Preconditioner< Matrix > >();
   pre->setup( parameters );
   solver.setPreconditioner( pre );
   // preconditioner update may throw if it's not implemented for CUDA
   try {
      pre->update( matrix );
   }
   catch( const std::runtime_error& ) {
   }
   catch( std::invalid_argument& e ) {
      std::cerr << e.what() << ". Skipping the benchmark!\n";
      return;
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
      const bool converged = solver.solve( b, x );
      barrier( matrix );
      if( ! converged )
         throw std::runtime_error( "solver did not converge" );
   };

   BenchmarkResult< Vector, Matrix, Solver > benchmarkResult( solver, matrix, x, b );
   benchmark.time< typename Matrix::DeviceType >( reset, performer, compute, benchmarkResult );
}

template< template< typename > class Solver, typename Matrix, typename Vector >
void
benchmarkDirectSolver( const TNL::String& solverName,
                       TNL::Benchmarks::Benchmark<>& benchmark,
                       const TNL::Config::ParameterContainer& parameters,
                       const std::shared_ptr< Matrix >& matrix,
                       const Vector& x0,
                       const Vector& b )
{
   // skip benchmarks on devices which the user did not select
   if( ! checkDevice< typename Matrix::DeviceType >( parameters ) )
      return;

   barrier( matrix );
   const char* performer = getPerformer< typename Matrix::DeviceType >();

   // setup
   Solver< Matrix > solver;
   solver.setup( parameters );

   Vector x( x0.getSize(), 0 );

   BenchmarkResult< Vector, Matrix, Solver > benchmarkResult( solver, matrix, x, b );

   auto set_matrix = [ & ]()
   {
      solver.setMatrix( matrix );
   };
   benchmark.setOperation( solverName + " setup" );
   benchmark.time< typename Matrix::DeviceType >( set_matrix, performer, set_matrix, benchmarkResult );

   solver.setSolverMonitor( benchmark.getMonitor() );

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
         throw std::runtime_error( "solver failed" );
   };
   benchmark.setOperation( solverName + " solve" );
   benchmark.time< typename Matrix::DeviceType >( reset, performer, compute, benchmarkResult );
}
