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
   Array< double, Devices::Host > host_input( 10 );
   Array< double, Devices::Host > host_output( 10 );
   host_input = 1.0;
   std::cout << "host_input = " << host_input << '\n';
   inclusiveScan( host_input, host_output );
   std::cout << "host_output " << host_output << '\n';

   /***
    * And then also on GPU.
    */
#if defined( __CUDACC__ ) || defined( __HIP__ )
   Array< double, Devices::GPU > gpu_input( 10 );
   Array< double, Devices::GPU > gpu_output( 10 );
   gpu_input = 1.0;
   std::cout << "gpu_input = " << gpu_input << '\n';
   inclusiveScan( gpu_input, gpu_output );
   std::cout << "gpu_output " << gpu_output << '\n';
#endif
   return EXIT_SUCCESS;
}
