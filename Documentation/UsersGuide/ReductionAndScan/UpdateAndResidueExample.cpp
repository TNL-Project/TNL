#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/reduce.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
double
updateAndResidue( Vector< double, Device >& u, const Vector< double, Device >& delta_u, const double& tau )
{
   auto u_view = u.getView();
   auto delta_u_view = delta_u.getConstView();
   auto fetch = [ = ] __cuda_callable__( int i ) mutable -> double
   {
      const double& add = delta_u_view[ i ];
      u_view[ i ] += tau * add;
      return add * add;
   };
   auto reduction = [] __cuda_callable__( const double& a, const double& b )
   {
      return a + b;
   };
   return sqrt( reduce< Device >( 0, u_view.getSize(), fetch, reduction, 0.0 ) );
}

int
main( int argc, char* argv[] )
{
   const double tau = 0.1;
   Vector< double, Devices::Host > host_u( 10 );
   Vector< double, Devices::Host > host_delta_u( 10 );
   host_u = 0.0;
   host_delta_u = 1.0;
   std::cout << "host_u = " << host_u << '\n';
   std::cout << "host_delta_u = " << host_delta_u << '\n';
   double residue = updateAndResidue( host_u, host_delta_u, tau );
   std::cout << "New host_u is: " << host_u << '\n';
   std::cout << "Residue is: " << residue << '\n';

#if defined( __CUDACC__ ) || defined( __HIP__ )
   Vector< double, Devices::GPU > gpu_u( 10 );
   Vector< double, Devices::GPU > gpu_delta_u( 10 );
   gpu_u = 0.0;
   gpu_delta_u = 1.0;
   std::cout << "gpu_u = " << gpu_u << '\n';
   std::cout << "gpu_delta_u = " << gpu_delta_u << '\n';
   residue = updateAndResidue( gpu_u, gpu_delta_u, tau );
   std::cout << "New gpu_u is: " << gpu_u << '\n';
   std::cout << "Residue is: " << residue << '\n';
#endif
   return EXIT_SUCCESS;
}
