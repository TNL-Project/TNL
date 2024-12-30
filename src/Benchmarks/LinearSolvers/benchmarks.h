// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Solvers/IterativeSolverMonitor.h>
#include <TNL/Matrices/DistributedMatrix.h>

#include <stdexcept>  // std::runtime_error
#include "BenchmarkResults.h"

#ifdef HAVE_ARMADILLO
   #include <armadillo>
   #include <TNL/Matrices/CSR.h>
#endif

template< typename Device >
const char*
getPerformer()
{
   if( std::is_same< Device, TNL::Devices::Cuda >::value )
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
   const TNL::String device = parameters.getParameter< TNL::String >( "devices" );
   if( device == "all" )
      return true;
   if( std::is_same< Device, TNL::Devices::Host >::value && device == "host" )
      return true;
   if( std::is_same< Device, TNL::Devices::Cuda >::value && device == "cuda" )
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

   // FIXME: getMonitor returns solver monitor specialized for double and int
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
      std::cerr << e.what() << ". Skipping the benchmark!" << std::endl;
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

   // FIXME: getMonitor returns solver monitor specialized for double and int
   //solver.setSolverMonitor( benchmark.getMonitor() ); // benchmark returns only IterativeSolverMonitor

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

#ifdef HAVE_ARMADILLO
// TODO: make a TNL solver like UmfpackWrapper
template< typename Vector >
void
benchmarkArmadillo( const Config::ParameterContainer& parameters,
                    const std::shared_ptr< Matrices::CSR< double, Devices::Host, int > >& matrix,
                    const Vector& x0,
                    const Vector& b )
{
   // copy matrix into Armadillo's class
   // sp_mat is in CSC format
   arma::uvec _colptr( matrix->getRowPointers().getSize() );
   for( int i = 0; i < matrix->getRowPointers().getSize(); i++ )
      _colptr[ i ] = matrix->getRowPointers()[ i ];
   arma::uvec _rowind( matrix->getColumnIndexes().getSize() );
   for( int i = 0; i < matrix->getColumnIndexes().getSize(); i++ )
      _rowind[ i ] = matrix->getColumnIndexes()[ i ];
   arma::vec _values( matrix->getValues().getData(), matrix->getValues().getSize() );
   arma::sp_mat AT( _rowind, _colptr, _values, matrix->getColumns(), matrix->getRows() );
   arma::sp_mat A = AT.t();

   Vector x;
   x.setLike( x0 );

   // Armadillo vector using the same memory as x (with copy_aux_mem=false, strict=true)
   arma::vec arma_x( x.getData(), x.getSize(), false, true );
   arma::vec arma_b( b.getData(), b.getSize() );

   arma::superlu_opts settings;
   //    settings.equilibrate = false;
   settings.equilibrate = true;
   settings.pivot_thresh = 1.0;
   //    settings.permutation = arma::superlu_opts::COLAMD;
   settings.permutation = arma::superlu_opts::MMD_AT_PLUS_A;
   //    settings.refine = arma::superlu_opts::REF_DOUBLE;
   settings.refine = arma::superlu_opts::REF_NONE;

   // reset function
   auto reset = [ & ]()
   {
      x = x0;
   };

   // benchmark function
   auto compute = [ & ]()
   {
      const bool converged = arma::spsolve( arma_x, A, arma_b, "superlu", settings );
      if( ! converged )
         throw std::runtime_error( "solver did not converge" );
   };

   const int loops = parameters.getParameter< int >( "loops" );
   double time = timeFunction( compute, reset, loops );

   arma::vec r = A * arma_x - arma_b;
   //    std::cout << "Converged: " << (time > 0) << ", residue = " << arma::norm( r ) / arma::norm( arma_b ) << " " <<
   //    std::endl; std::cout << "Mean time: " << time / loops << " seconds." << std::endl;
   std::cout << "Converged: " << std::setw( 5 ) << std::boolalpha << ( time > 0 ) << "   " << "iterations = " << std::setw( 4 )
             << "N/A" << "   " << "residue = " << std::setw( 10 ) << arma::norm( r ) / arma::norm( arma_b ) << "   "
             << "mean time = " << std::setw( 9 ) << time / loops << " seconds." << std::endl;
}
#endif
