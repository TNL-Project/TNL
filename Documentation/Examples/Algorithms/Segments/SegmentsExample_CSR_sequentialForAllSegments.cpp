#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/traverse.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
SegmentsExample()
{
   using SegmentsType = typename TNL::Algorithms::Segments::CSR< Device, int >;
   using SegmentView = typename SegmentsType::SegmentViewType;

   /***
    * Create segments with given segments sizes.
    */
   SegmentsType segments{ 1, 2, 3, 4, 5 };
   std::cout << "Segments sizes are: " << segments << '\n';

   /***
    * Print the elements mapping using segment view.
    */
   std::cout << "Elements mapping:" << std::endl;
   TNL::Algorithms::Segments::sequentialForAllSegments(
      segments,
      [] __cuda_callable__( const SegmentView segment )
      {
         printf( "Segment idx. %d: \n", segment.getSegmentIndex() );  // printf works even in GPU kernels
         for( auto element : segment )
            printf( "%d -> %d  ", element.localIndex(), element.globalIndex() );
      } );
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
