#include <iostream>
#include <cstdlib>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/reduce.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
double
product( const Vector< double, Device >& v )
{
   auto view = v.getConstView();
   auto fetch = [ = ] __cuda_callable__( int i )
   {
      return view[ i ];
   };
   auto reduction = [] __cuda_callable__( const double& a, const double& b )
   {
      return a * b;
   };

   /***
    * Since we compute the product of all elements, the reduction must be initialized by 1.0 not by 0.0.
    */
   return reduce< Device >( 0, view.getSize(), fetch, reduction, 1.0 );
}

int
main( int argc, char* argv[] )
{
   /***
    * The first test on CPU ...
    */
   Vector< double, Devices::Host > host_v( 10 );
   host_v = 1.0;
   std::cout << "host_v = " << host_v << '\n';
   std::cout << "The product of the host vector elements is " << product( host_v ) << ".\n";

   /***
    * ... the second test on GPU.
    */
#if defined( __CUDACC__ ) || defined( __HIP__ )
   Vector< double, Devices::GPU > gpu_v( 10 );
   gpu_v = 1.0;
   std::cout << "gpu_v = " << gpu_v << '\n';
   std::cout << "The product of the GPU vector elements is " << product( gpu_v ) << ".\n";
#endif
   return EXIT_SUCCESS;
}
