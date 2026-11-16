#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/reduce.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
bool
comparison( const Vector< double, Device >& u, const Vector< double, Device >& v )
{
   auto u_view = u.getConstView();
   auto v_view = v.getConstView();

   /***
    * Fetch compares corresponding elements of both vectors
    */
   auto fetch = [ = ] __cuda_callable__( int i ) -> bool
   {
      return u_view[ i ] == v_view[ i ];
   };

   /***
    * Reduce performs logical AND on intermediate results obtained by fetch.
    */
   auto reduction = [] __cuda_callable__( const bool& a, const bool& b )
   {
      return a && b;
   };
   return reduce< Device >( 0, v_view.getSize(), fetch, reduction, true );
}

int
main( int argc, char* argv[] )
{
   Vector< double, Devices::Host > host_u( 10 );
   Vector< double, Devices::Host > host_v( 10 );
   host_u = 1.0;
   host_v.forAllElements(
      [] __cuda_callable__( int i, double& value )
      {
         value = 2 * ( i % 2 ) - 1;
      } );
   std::cout << "host_u = " << host_u << '\n';
   std::cout << "host_v = " << host_v << '\n';
   std::cout << "Comparison of host_u and host_v is: " << ( comparison( host_u, host_v ) ? "'true'" : "'false'" ) << ".\n";
   std::cout << "Comparison of host_u and host_u is: " << ( comparison( host_u, host_u ) ? "'true'" : "'false'" ) << ".\n";

#if defined( __CUDACC__ ) || defined( __HIP__ )
   Vector< double, Devices::GPU > gpu_u( 10 ), gpu_v( 10 );
   gpu_u = 1.0;
   gpu_v.forAllElements(
      [] __cuda_callable__( int i, double& value )
      {
         value = 2 * ( i % 2 ) - 1;
      } );
   std::cout << "gpu_u = " << gpu_u << '\n';
   std::cout << "gpu_v = " << gpu_v << '\n';
   std::cout << "Comparison of gpu_u and gpu_v is: " << ( comparison( gpu_u, gpu_v ) ? "'true'" : "'false'" ) << ".\n";
   std::cout << "Comparison of gpu_u and gpu_u is: " << ( comparison( gpu_u, gpu_u ) ? "'true'" : "'false'" ) << ".\n";
#endif
   return EXIT_SUCCESS;
}
