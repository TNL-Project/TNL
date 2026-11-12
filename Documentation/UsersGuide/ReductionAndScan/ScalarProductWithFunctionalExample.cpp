#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/reduce.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
double
scalarProduct( const Vector< double, Device >& u, const Vector< double, Device >& v )
{
   auto u_view = u.getConstView();
   auto v_view = v.getConstView();

   /***
    * Fetch computes product of corresponding elements of both vectors.
    */
   return reduce< Device >(
      0,
      v_view.getSize(),
      [ = ] __cuda_callable__( int i )
      {
         return u_view[ i ] * v_view[ i ];
      },
      TNL::Plus{} );
}

int
main( int argc, char* argv[] )
{
   /***
    * The first test on CPU ...
    */
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
   std::cout << "The scalar product ( host_u, host_v ) is " << scalarProduct( host_u, host_v ) << ".\n";
   std::cout << "The scalar product ( host_v, host_v ) is " << scalarProduct( host_v, host_v ) << ".\n";

   /***
    * ... the second test on GPU.
    */
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
   std::cout << "The scalar product ( gpu_u, gpu_v ) is " << scalarProduct( gpu_u, gpu_v ) << ".\n";
   std::cout << "The scalar product ( gpu_v, gpu_v ) is " << scalarProduct( gpu_v, gpu_v ) << ".\n";
#endif
   return EXIT_SUCCESS;
}
