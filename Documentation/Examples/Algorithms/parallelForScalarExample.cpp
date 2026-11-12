#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/parallelFor.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

/****
 * Set all elements of the vector v to the constant c.
 */
template< typename Device >
void
initVector( Vector< double, Device >& v, const double& c )
{
   auto view = v.getView();
   auto init = [ = ] __cuda_callable__( int i ) mutable
   {
      view[ i ] = c;
   };
   parallelFor< Device >( 0, v.getSize(), init );
}

int
main( int argc, char* argv[] )
{
   /***
    * Firstly, test the vector initiation on CPU.
    */
   Vector< double, Devices::Host > host_v( 10 );
   initVector( host_v, 1.0 );
   std::cout << "host_v = " << host_v << '\n';

   /***
    * And then also on GPU.
    */
#if defined( __CUDACC__ ) || defined( __HIP__ )
   Vector< double, Devices::GPU > gpu_v( 10 );
   initVector( gpu_v, 1.0 );
   std::cout << "gpu_v = " << gpu_v << '\n';
#endif
   return EXIT_SUCCESS;
}
