#include <iostream>
#include <TNL/Containers/BlockPartitioning.h>
#include <TNL/Containers/DistributedArraySynchronizer.h>
#include <TNL/Containers/DistributedVector.h>
#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Euler.h>
#include <string>
#include "write.h"

using Real = double;
using Index = int;

template< typename Device >
void
solveHeatEquation( const char* file_name )
{
   //! [Types definition]
   using Vector = TNL::Containers::DistributedVector< Real, Device, Index >;
   using VectorView = typename Vector::ViewType;
   using LocalVectorView = typename Vector::LocalViewType;
   using ConstLocalVectorView = typename Vector::ConstLocalViewType;
   using LocalRangeType = typename Vector::LocalRangeType;
   using Synchronizer = TNL::Containers::DistributedArraySynchronizer< Vector >;
   using Method = TNL::Solvers::ODE::Methods::Euler< Real >;
   using ODESolver = TNL::Solvers::ODE::ODESolver< Method, Vector >;
   //! [Types definition]

   //! [Parameters of the discretisation]
   const Real final_t = 0.05;
   const Real output_time_step = 0.005;
   const Index n = 41;
   const Real h = 1.0 / ( n - 1 );
   const Real tau = 0.1 * h * h;
   const Real h_sqr_inv = 1.0 / ( h * h );
   //! [Parameters of the discretisation]

   //! [Domain decomposition]
   const TNL::MPI::Comm communicator = MPI_COMM_WORLD;
   const LocalRangeType localRange = TNL::Containers::splitRange< Index >( n, communicator );
   const Index ghosts = 2;

   Vector u( localRange, ghosts, n, communicator );
   u.setSynchronizer( std::make_shared< Synchronizer >( localRange, ghosts / 2, communicator ) );
   //! [Domain decomposition]

   //! [Initial condition]
   u.forElements( 0,
                  n,
                  [ = ] __cuda_callable__( Index i, Real & value )
                  {
                     const Real x = i * h;
                     if( x >= 0.4 && x <= 0.6 )
                        value = 1.0;
                     else
                        value = 0.0;
                  } );
   std::fstream file;
   file.open( file_name, std::ios::out );
   write( file, u, h, (Real) 0.0 );
   //! [Initial condition]

   //! [Solver setup]
   ODESolver solver;
   solver.setTau( tau );
   solver.setTime( 0.0 );
   //! [Solver setup]

   //! [Time loop]
   while( solver.getTime() < final_t ) {
      solver.setStopTime( TNL::min( solver.getTime() + output_time_step, final_t ) );
      //! [Lambda function f]
      auto f = [ = ] __cuda_callable__( Index gi, const ConstLocalVectorView& u, LocalVectorView& fu )
      {
         // local index of the current node
         const Index i = localRange.getLocalIndex( gi );

         if( gi == 0 || gi == n - 1 ) {
            // boundary nodes -> boundary conditions
            fu[ i ] = 0.0;
         }
         else {
            // local indices of the left and right neighbors
            const Index left = localRange.isLocal( gi - 1 ) ? i - 1 : localRange.getSize();
            const Index right = localRange.isLocal( gi + 1 ) ? i + 1 : localRange.getSize() + 1;

            // interior nodes -> approximation of the second derivative
            fu[ i ] = h_sqr_inv * ( u[ left ] - 2.0 * u[ i ] + u[ right ] );
         }
      };
      //! [Lambda function f]
      //! [Lambda function time_stepping]
      auto time_stepping = [ = ]( const Real& t, const Real& tau, const VectorView& u, VectorView& fu ) mutable
      {
         //! [Parallel for call]
         const_cast< VectorView& >( u ).startSynchronization();
         u.waitForSynchronization();
         TNL::Algorithms::parallelFor< Device >(
            localRange.getBegin(), localRange.getEnd(), f, u.getConstLocalViewWithGhosts(), fu.getLocalView() );
         //! [Parallel for call]
      };
      //! [Lambda function time_stepping]
      solver.solve( u, time_stepping );
      write( file, u, h, solver.getTime() );  // write the current state to a file
   }
   //! [Time loop]
}

int
main( int argc, char* argv[] )
{
   TNL::MPI::ScopedInitializer mpi( argc, argv );

   if( argc != 2 ) {
      std::cerr << "Usage: " << argv[ 0 ] << " <path to output directory>\n";
      return EXIT_FAILURE;
   }
   TNL::String file_name( argv[ 1 ] );
   file_name += "/DistributedODESolver-HeatEquationExample-result-";
   file_name += std::to_string( TNL::MPI::GetRank() );
   file_name += ".out";

   solveHeatEquation< TNL::Devices::Host >( file_name.getString() );
#ifdef __CUDACC__
   TNL::MPI::Barrier();
   solveHeatEquation< TNL::Devices::Cuda >( file_name.getString() );
#endif
}
