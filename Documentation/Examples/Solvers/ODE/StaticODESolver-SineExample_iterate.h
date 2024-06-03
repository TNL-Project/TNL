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
   Real time = 0.0;
   Real current_tau = tau;
   //! [Time variables]

   //! [Solver setup]
   Vector u = 0.0;
   ODESolver solver;
   solver.init( u );
   //! [Solver setup]

   //! [Time loop]
   while( time < final_time ) {
      auto f = []( const Real& t, const Real& current_tau, const Vector& u, Vector& fu )
      {
         fu = t * sin( t );
      };
      solver.iterate( u, time, current_tau, f );
      if( time >= next_output_time ) {
         std::cout << time << " " << u[ 0 ] << std::endl;
         next_output_time += output_time_step;
         if( next_output_time > final_time )
            next_output_time = final_time;
      }
      if( time + current_tau > next_output_time )
         current_tau = next_output_time - time;
      else
         current_tau = tau;
   }
   //! [Time loop]
}
//! [Main function]
