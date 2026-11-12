#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/scan.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

int
main( int argc, char* argv[] )
{
   /***
    * Firstly, test the prefix sum with an array allocated on CPU.
    */
   Array< double, Devices::Host > host_a( 10 );
   host_a = 1.0;
   std::cout << "host_a = " << host_a << '\n';
   inplaceInclusiveScan( host_a );
   std::cout << "The prefix sum of the host array is " << host_a << ".\n";

   /***
    * And then also on GPU.
    */
#if defined( __CUDACC__ ) || defined( __HIP__ )
   Array< double, Devices::GPU > gpu_a( 10 );
   gpu_a = 1.0;
   std::cout << "gpu_a = " << gpu_a << '\n';
   inplaceInclusiveScan( gpu_a );
   std::cout << "The prefix sum of the GPU array is " << gpu_a << ".\n";
#endif
   return EXIT_SUCCESS;
}
