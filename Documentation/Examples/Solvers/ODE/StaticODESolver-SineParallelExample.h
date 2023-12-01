#include <iostream>
#include <fstream>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Euler.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/parallelFor.h>

using Real = double;

template< typename Device >
void
solveParallelODEs( const char* file_name )
{
   using StaticVector = TNL::Containers::StaticVector< 1, Real >;
   using Vector = TNL::Containers::Vector< StaticVector, Device >;
   using Method = TNL::Solvers::ODE::Methods::Euler<>;
   using ODESolver = TNL::Solvers::ODE::ODESolver< Method, StaticVector >;

   const Real final_t = 10.0;
   const Real tau = 0.001;
   const Real output_time_step = 0.1;
   const Real c_min = 1.0;
   const Real c_max = 5.0;
   const int c_vals = 5.0;
   const Real c_step = ( c_max - c_min ) / ( c_vals - 1 );
   const int output_time_steps = ceil( final_t / output_time_step ) + 1;

   Vector results( output_time_steps * c_vals, 0.0 );
   auto results_view = results.getView();
   auto f = [ = ] __cuda_callable__( const Real& t, const Real& tau, const StaticVector& u, StaticVector& fu, const Real& c )
   {
      fu = t * sin( c * t );
   };
   auto solve = [ = ] __cuda_callable__( int idx ) mutable
   {
      const Real c = c_min + idx * c_step;
      ODESolver solver;
      solver.setTau( tau );
      solver.setTime( 0.0 );
      StaticVector u = 0.0;
      int time_step( 1 );
      results_view[ idx ] = u;
      while( time_step < output_time_steps ) {
         solver.setStopTime( TNL::min( solver.getTime() + output_time_step, final_t ) );
         solver.solve( u, f, c );
         results_view[ time_step++ * c_vals + idx ] = u;
      }
   };
   TNL::Algorithms::parallelFor< Device >( 0, c_vals, solve );

   std::fstream file;
   file.open( file_name, std::ios::out );
   for( int i = 0; i < c_vals; i++ ) {
      file << "# c = " << c_min + i * c_step << std::endl;
      for( int k = 0; k < output_time_steps; k++ )
         file << k * output_time_step << " " << results.getElement( k * c_vals + i )[ i ] << std::endl;
      file << std::endl;
   }
}

int
main( int argc, char* argv[] )
{
   TNL::String file_name( argv[ 1 ] );
   file_name += "/StaticODESolver-SineParallelExample-result.out";
   solveParallelODEs< TNL::Devices::Host >( file_name.getString() );
#ifdef __CUDACC__
   solveParallelODEs< TNL::Devices::Cuda >( file_name.getString() );
#endif
}
