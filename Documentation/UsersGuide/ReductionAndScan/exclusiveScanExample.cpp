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
   exclusiveScan( host_input, host_output );
   std::cout << "host_output " << host_output << '\n';

   /***
    * And then also on GPU.
    */
#ifdef __CUDACC__
   Array< double, Devices::Cuda > cuda_input( 10 );
   Array< double, Devices::Cuda > cuda_output( 10 );
   cuda_input = 1.0;
   std::cout << "cuda_input = " << cuda_input << '\n';
   exclusiveScan( cuda_input, cuda_output );
   std::cout << "cuda_output " << cuda_output << '\n';
#endif
   return EXIT_SUCCESS;
}
