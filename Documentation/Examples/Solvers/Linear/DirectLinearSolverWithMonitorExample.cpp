#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Solvers/Linear/UmfpackWrapper.h>
#include <TNL/Solvers/DirectSolverMonitor.h>

template< typename Device >
void
directLinearSolverExample()
{
   /***
    * Set the following matrix (dots represent zero matrix elements):
    *
    *   /  2.5 -1    .    .    .   \
    *   | -1    2.5 -1    .    .   |
    *   |  .   -1    2.5 -1.   .   |
    *   |  .    .   -1    2.5 -1   |
    *   \  .    .    .   -1    2.5 /
    */
   using MatrixType = TNL::Matrices::SparseMatrix< double, Device >;
   using Vector = TNL::Containers::Vector< double, Device >;
   const int size( 5 );
   auto matrix_ptr = std::make_shared< MatrixType >();
   matrix_ptr->setDimensions( size, size );
   matrix_ptr->setRowCapacities( Vector( { 2, 3, 3, 3, 2 } ) );

   auto f = [ = ] __cuda_callable__( typename MatrixType::RowView & row ) mutable
   {
      const int rowIdx = row.getRowIndex();
      if( rowIdx == 0 ) {
         row.setElement( 0, rowIdx, 2.5 );     // diagonal element
         row.setElement( 1, rowIdx + 1, -1 );  // element above the diagonal
      }
      else if( rowIdx == size - 1 ) {
         row.setElement( 0, rowIdx - 1, -1.0 );  // element below the diagonal
         row.setElement( 1, rowIdx, 2.5 );       // diagonal element
      }
      else {
         row.setElement( 0, rowIdx - 1, -1.0 );  // element below the diagonal
         row.setElement( 1, rowIdx, 2.5 );       // diagonal element
         row.setElement( 2, rowIdx + 1, -1.0 );  // element above the diagonal
      }
   };

   /***
    * Set the matrix elements.
    */
   matrix_ptr->forAllRows( f );
   std::cout << *matrix_ptr << std::endl;

   /***
    * Set the right-hand side vector.
    */
   Vector x( size, 1.0 );
   Vector b( size );
   matrix_ptr->vectorProduct( x, b );
   x = 0.0;
   std::cout << "Vector b = " << b << std::endl;

   /***
    * Setup solver of the linear system.
    */
   using LinearSolver = TNL::Solvers::Linear::UmfpackWrapper< MatrixType >;
   LinearSolver solver;
   solver.setMatrix( matrix_ptr );

   /***
    * Setup monitor of the iterative solver.
    */
   using DirectSolverMonitorType = TNL::Solvers::DirectSolverMonitor< double, int >;
   DirectSolverMonitorType monitor;
   TNL::Solvers::SolverMonitorThread monitorThread( monitor );
   monitor.setRefreshRate( 10 );  // refresh rate in milliseconds
   monitor.setVerbose( 1 );
   monitor.setStage( "Umfpack stage:" );
   solver.setSolverMonitor( monitor );
   solver.solve( b, x );
   monitor.stopMainLoop();
   if( solver.succeeded() ) {
      std::cout << "Solver succeeded." << std::endl;
      std::cout << "Vector x = " << x << std::endl;
   }
   else
      std::cout << "Solver failed." << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Solving linear system on host: " << std::endl;
   directLinearSolverExample< TNL::Devices::Sequential >();
}
