#include <iostream>
#include <TNL/Solvers/ODE/ODESolver.h>
#include <TNL/Solvers/ODE/Methods/Euler.h>

using Real = double;

int
main( int argc, char* argv[] )
{
   using Vector = TNL::Containers::StaticVector< 1, Real >;
   using Method = TNL::Solvers::ODE::Methods::Euler< Real >;
   using ODESolver = TNL::Solvers::ODE::ODESolver< Method, Vector >;

   const Real final_t = 10.0;
   const Real tau = 0.001;
   const Real output_time_step = 0.25;

   ODESolver solver;
   solver.setTau( tau );
   solver.setTime( 0.0 );
   Vector u = 0.0;
   while( solver.getTime() < final_t ) {
      solver.setStopTime( TNL::min( solver.getTime() + output_time_step, final_t ) );
      auto f = []( const Real& t, const Real& tau, const Vector& u, Vector& fu )
      {
         fu = t * sin( t );
      };
      solver.solve( u, f );
      std::cout << solver.getTime() << " " << u[ 0 ] << std::endl;
   }
}
