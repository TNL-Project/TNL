#include <iostream>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
SegmentsExample()
{
   using SegmentsType = typename TNL::Algorithms::Segments::CSR< Device, int >;

   /***
    * Create segments and print the segments type.
    */
   SegmentsType segments;
   std::cout << "The segments type is: " << segments.getSegmentsType() << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Example of CSR segments on host:\n";
   SegmentsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Example of CSR segments on CUDA GPU:\n";
   SegmentsExample< TNL::Devices::Cuda >();
#endif
   return EXIT_SUCCESS;
}
