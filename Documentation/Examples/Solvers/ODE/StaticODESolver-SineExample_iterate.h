#include <algorithm>
#include <iostream>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Euler.h>

//! [Real definition]
using Real = double;
//! [Real definition]

//! [Main function]
int
main( int argc, char* argv[] )
{
   //! [Types definition]
   using Vector = TNL::Containers::StaticVector< 1, Real >;
   using Method = TNL::Solvers::ODE::Methods::Euler< Real >;
   using ODESolver = TNL::Solvers::ODE::ODESolver< Method, Vector >;
   //! [Types definition]

   //! [Time variables]
   const Real final_time = 10.0;
   const Real output_time_step = 0.25;
   Real next_output_time = TNL::min( output_time_step, final_time );
   Real tau = 0.001;
   //! [Time variables]

   //! [Solver setup]
   Vector u = 0.0;
   ODESolver solver;
   solver.init( u );
   solver.setTime( 0 );
   solver.setStopTime( final_time );
   solver.setTau( tau );
   //! [Solver setup]

   //! [Time loop]
   while( solver.getTime() < final_time ) {
      auto f = []( const Real& t, const Real& current_tau, const Vector& u, Vector& fu )
      {
         fu = t * sin( t );
      };
      solver.iterate( u, f );
      if( solver.getTime() >= next_output_time ) {
         std::cout << solver.getTime() << " " << u[ 0 ] << '\n';
         next_output_time += output_time_step;
         next_output_time = std::min( next_output_time, final_time );
      }
   }
   //! [Time loop]
}
//! [Main function]
